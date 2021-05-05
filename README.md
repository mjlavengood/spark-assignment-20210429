# Sayari Spark Assignment

### How to Run

The script can be run by cd'ing into the repository root directory and running following command (tested with Python 3.6.9 and Spark 3.1.1). Unfortunately it still takes around 2 hours to run locally on my laptop, but I included the results in the repo under `result/` in case you just want to look directly at those.

```
spark-submit  --packages com.databricks:spark-xml_2.12:0.12.0  --master local script.py
```

### Background info and explanation

The goal of this exercise is to identify entities appearing in both the UK Treasury sanctions data (ConList.csv) and the US sanctions data from OFAC (sdn.xml).

Due to time constraints and the difficulty of working with this heterogenous data, I ended up using only the Name fields and the date of birth fields in the respective data - "DOB" and "Name 1" - "Name 6" in the UK data, and "dateOfBirthList", "firstName", and "lastName" (main and AKAs) in the US data.

Since the fuzzy string comparison is rather performance-intensive, I initially do a birth year comparison to narrow down potetial matches from a full Cartesian join. I use year rather than date because the exact birthdates seem inconsistent, and many records only include the year or approximate year anyway. I use regex to extract year information from the given birthdates. Since the US sometimes includes ranges instead of single dates, I calculate a min and max year for each US birthdate entry, and match it to UK records with a birth year between the min and max date. I then calculate pairs of records for which there is a birth year match, or for which no birth year data is available for one or both records (so that companies, vessels, and persons with incomplete data are still kept as candidates for matching).

The birth-year filter seems to be effective to eliminate many erroneous matches on individuals (e.g. families where members have highly similar full names but different birth years). Obviously it does not help with matching companies or vessels - that would be a task for future development.

After generating a list of match candidates based on birth year, I concatenate together all name fields in the UK and US data to form full names. I perform a fuzzy comparison between the UK and US full names using TF-IDF and cosine similarity. (This unfortunately takes a few hours on my laptop - I didn't have time to optimize performance further!) I chose 0.7 cosine similarity as the threshold for a match, somewhat arbitrarily on the basis of observations of when a score tended to yield a true match.

My result file has 1193 matches with the following schema:

```
uk_id: string (UK Sanctions List Ref extracted from the "Other Information" field)
us_id: long ("uid" field from US data)
entity_type: string ("Individual", "Entity", or "Vessel" from "sdnType" in US data)
uk_birth_years: array[string] (UK birth years for which matches were found with US data)
us_birth_year_ranges: array[array[string]] (US birth year ranges for which matches were found with UK data)
fullname_similarity: double (max TF-IDF cosine similarity score between UK and US fullnames)
full_name_uk: array[string] (UK full name strings used for comparison)
full_name_us: array[string] (US full name strings used for comparison)
```

Due to time constraints, I decided to stop here. Future improvement points would include verifying company matching accuracy, improving the performance of the fuzzy string match (probably by using other fields to make more narrow initial preliminary matches), and general code improvement (unit tests, refactoring, repo structure etc). Down the line, it would probably best to use a more holistic probabilistic model which considers all fields simultaneously, weighted by importance, to make a general match score for all candidate record pairs.
