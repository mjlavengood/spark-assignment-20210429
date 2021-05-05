from pyspark.ml.feature import IDF, HashingTF, NGram, Normalizer
from pyspark.sql import SparkSession, DataFrame, Column
from pyspark.sql import functions as func
from pyspark.sql.types import DoubleType, LongType
from typing import Union

US_DATA_PATH = "sdn.xml"
UK_DATA_PATH = "ConList.csv"
UK_DATA_PATH_DATE_HEADER_REMOVED = "ConList_header_removed.csv"

TF_IDF_SIMILARITY_THRESHOLD = 0.7

TARGET_PATH = "result"


def concat_and_clean(*columns: Union[str, Column]) -> Column:
    """
    :param columns: columns to concatenate in order
    :type columns: List[str] or List[Column]
    :return: a new column concatenating the colums, all lowercase and cleaned
    """
    return func.regexp_replace(func.lower(func.concat_ws("", *columns)), "[^a-z]", "")


def load_us_data(spark: SparkSession) -> DataFrame:
    """
    Load the OFAC data from a CSV.
    :return: a dataframe with the OFAC data, plus full_name_us and us_id
    """
    return (
        spark.read.format("com.databricks.spark.xml")
        .option("rootTag", "sdnList")
        .option("rowTag", "sdnEntry")
        .load(US_DATA_PATH)
        .withColumn("us_id", func.col("uid"))
    )


def load_uk_data(spark: SparkSession) -> DataFrame:
    """
    Load the UK Treasury data from a CSV.

    :return: a dataframe with the UK treasury data, plus full_name_uk and uk_id
    """
    remove_first_line_from_csv(UK_DATA_PATH, UK_DATA_PATH_DATE_HEADER_REMOVED)
    return (
        spark.read.csv(path=UK_DATA_PATH_DATE_HEADER_REMOVED, header=True)
        # extract UK Sanctions List ID from "other information" column
        .withColumn(
            "uk_id",
            func.regexp_extract(
                func.col("Other Information"), "\(UK Sanctions List Ref\):([^\s,]*)", 1
            ),
        ).where(func.col("uk_id").isNotNull())
    )


def remove_first_line_from_csv(path: str, new_path: str) -> None:
    """
    Remove the first line of the csv (UK data has a date header)

    :param path: original file path
    :param new_path: where to store the truncated file
    """
    with open(path, "rb") as f:
        with open(new_path, "wb") as f1:
            f.readline()
            for line in f:
                f1.write(line)


def get_fullname_matches(
    df_matches: DataFrame, df_us: DataFrame, df_uk: DataFrame
) -> DataFrame:
    """
    Find matches in US/UK data based on TF-IDF similarity of
    their full names

    :param df_matches: dataframe of existing matches found so far
    :param df_us: US data
    :param df_uk: UK data
    """

    def add_tf(df, input_col, output_col):
        df = df.withColumn("characters", func.split(input_col, ""))
        ngram = NGram(n=3, inputCol="characters", outputCol="ngrams")
        df = ngram.transform(df)
        hashingTF = HashingTF(inputCol="ngrams", outputCol=output_col)
        return hashingTF.transform(df)

    def get_idf_model(df_uk, df_us, input_col, output_col):
        df_union = df_uk.select("tf").unionAll(df_us.select("tf"))
        idf = IDF(inputCol=input_col, outputCol=output_col)
        idfModel = idf.fit(df_union)
        return idfModel

    def get_normalized_idf(df, idfModel, output_col):
        df = idfModel.transform(df)
        normalizer = Normalizer(inputCol=idfModel.getOutputCol(), outputCol=output_col)
        return normalizer.transform(df)

    # Get US full names from firstName/lastName fields in root, as well as AKA
    # list
    #
    # First union primary name to AKA array, then explode, extract and
    # concatenate/clean first/last names
    df_us = (
        df_us.withColumn(
            "name_struct",
            func.explode(
                func.array_union(
                    func.array(
                        func.struct(
                            func.lit("").alias("category"),
                            func.col("firstName").alias("firstName"),
                            func.col("lastName").alias("lastName"),
                            func.lit("").alias("type"),
                            func.lit(0).cast(LongType()).alias("uid"),
                        )
                    ),
                    func.col("akaList.aka"),
                )
            ),
        )
        .withColumn(
            "full_name_us",
            concat_and_clean("name_struct.firstName", "name_struct.lastName"),
        )
        .select("us_id", "full_name_us", "sdnType")
        .distinct()
    )

    # Get UK full names by concatenating Names 1-6
    df_uk = (
        df_uk.withColumn(
            "full_name_uk",
            concat_and_clean(
                "Name 1", "Name 2", "Name 3", "Name 4", "Name 5", "Name 6",
            ),
        )
        .select("uk_id", "full_name_uk")
        .distinct()
    )

    # Calculate TF-IDF values for all full names. Build IDF model with unioned
    # data
    df_uk = add_tf(df_uk, "full_name_uk", "tf")
    df_us = add_tf(df_us, "full_name_us", "tf")

    idfModel = get_idf_model(df_uk, df_us, "tf", "idf")

    df_uk = get_normalized_idf(df_uk, idfModel, "normalized_idf_uk")
    df_us = get_normalized_idf(df_us, idfModel, "normalized_idf_us")

    dot_product_udf = func.udf(lambda x, y: float(x.dot(y)), DoubleType())

    # Filter match candidates by cosine similarity on full names
    #
    # Get max fullname similarity for UK-ID/US-ID pairs and filter by threshold
    return (
        df_matches.join(df_uk, on="uk_id")
        .join(df_us, on="us_id")
        .withColumn(
            "fullname_similarity",
            dot_product_udf("normalized_idf_uk", "normalized_idf_us"),
        )
        .where(func.col("fullname_similarity") > TF_IDF_SIMILARITY_THRESHOLD)
        .groupBy("uk_id", "us_id",)
        .agg(
            func.first("sdnType").alias("entity_type"),
            func.first("uk_birth_years").alias("uk_birth_years"),
            func.first("us_birth_year_ranges").alias("us_birth_year_ranges"),
            func.max(func.col("fullname_similarity")).alias("fullname_similarity"),
            func.collect_list("full_name_uk").alias("full_name_uk"),
            func.collect_list("full_name_us").alias("full_name_us"),
        )
    )


def get_candidates_by_birthyear(df_us: DataFrame, df_uk: DataFrame) -> DataFrame:
    """
    Find all match candidates in df_us and df_uk using birth year data.

    Use years instead of dates since exact date data is often fuzzy/possibly
    inaccurate.

    Exclude pairs for which no match can be found in any of the birthdates
    indicated in US/UK data. Do not exclude records where one or both
    birthdates are unknown (null)

    Include info on extracted birthyears in output data

    :param df_us: OFAC data
    :param df_uk: UK data
    :return a dataframe with UK_ID and US_ID potential matches,
            with info on birth years used for matching
    """
    # get US ID - DOB pairs
    df_us_dobs = (
        df_us.select(
            func.col("us_id"),
            func.explode_outer(df_us.dateOfBirthList.dateOfBirthItem.dateOfBirth).alias(
                "dob_us"
            ),
        )
        # extract min/max birth year from each indicated dob_us (some are ranges)
        .withColumn(
            "birth_years_us",
            func.array_remove(
                func.split(func.regexp_replace("dob_us", ".*?([0-9]{4})", "$1,"), ","),
                "",
            ),
        )
        .withColumn("min_birth_year_us", func.array_min("birth_years_us"))
        .withColumn("max_birth_year_us", func.array_max("birth_years_us"))
    )

    # get UK IDs with year of birth
    df_uk_dobs = df_uk.withColumn(
        "birth_year_uk", func.regexp_extract(func.col("DOB"), "([0-9]{4})", 1)
    )

    # join pairs and determine whether a birth year match exists in the US/UK data
    df_birthyear_matches = (
        df_us_dobs.join(
            df_uk_dobs,
            on=func.col("birth_year_uk").isNull()
            | func.col("min_birth_year_us").isNull()
            | func.col("birth_year_uk").between(
                func.col("min_birth_year_us"), func.col("max_birth_year_us")
            ),
        )
        .groupBy("uk_id", "us_id")
        .agg(
            func.collect_set("birth_year_uk").alias("uk_birth_years"),
            func.collect_set(
                func.array(func.col("min_birth_year_us"), func.col("max_birth_year_us"))
            ).alias("us_birth_year_ranges"),
        )
    )

    # return list with pairs excluded that only have mismatched birth years
    return df_birthyear_matches


if __name__ == "__main__":
    spark = SparkSession.builder.enableHiveSupport().getOrCreate()

    df_us = load_us_data(spark)
    df_uk = load_uk_data(spark)

    df_birthyear_matches = get_candidates_by_birthyear(df_us, df_uk)

    df_result = get_fullname_matches(df_birthyear_matches, df_us, df_uk)

    df_result.coalesce(10).write.mode("overwrite").parquet(TARGET_PATH)
