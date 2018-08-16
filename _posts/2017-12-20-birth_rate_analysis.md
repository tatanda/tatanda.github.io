---
title: "U.S Birth Rate Analysis"
data: 2017-12-20
tags: [Python, Pandas, data science]
header:
  image: ""
  excerpt: "U.S Birth Rate Analysis Excerpt"

---

<h1>Birth Rates in the United States</h1>  
<h4>This project is an analysis of a dataset of U.S birthrates from CDC's National Center for Health Statistics.</h4>

<p>The Dataset has 3653 rows of data and is organised with a header row with the following column variables;</p>

- <p>year : Year (1994 to 2003)</p>
- <p>month : Month (1 to 12)</p>
- <p>date_of_month : Day number of the month (1 to 31)</p>
- <p>day_of_week : Day of week (1 to 7)</p>
- <p>births : Number of births that day</p>



```python
f = open("births.csv", "r")
text = f.read()
# text = text.split("\n")
# text
```


```python
# new_txt = text.split("\n")
# print(len(new_txt))
```

<p>1. Read in datafile and Print data</p>


```python
def read_csv(filename):
    """
    input:
        filename: a csv file
    output:
        returns a nested list of data in csv file
    """
    with open(filename, "r") as f:
        string_list = f.read().split("\n")[1:]
    final_list = []
    for string in string_list:
        int_fields = list(map(lambda a : int(a), string.split(",")))
        final_list.append(int_fields)
    return final_list

def print_table(table):
    for row in table[:5]:
        for col in row:
            print("{:>12}".format(col), end="")
            #Remaining column right justified
        print("", end="\n")
header = ["year","month","date_of_month","day_of_week","births"]
print("{:>12} {:>12} {:>13} {:>10} {:>9}".format(header[0],header[1],header[2],header[3],header[4]))
print_table(read_csv("births.csv"))
```

            year        month date_of_month day_of_week    births
            1994           1           1           6        8096
            1994           1           2           7        7772
            1994           1           3           1       10142
            1994           1           4           2       11248
            1994           1           5           3       11053




<p>2. Aggregate number of births by column. i.e total births by month, year, day_of_week, day_of_month</p>


```python
def month_births(nested_list):
    """
    input:
        nested_list - list of list containing output birth data
        where each list consist of year, month, day of month
        day of week, and births for that day as elements
    Output:
        returns a dictionary of months as keys and monthly birth
        totals for the entire dataset as values
    """
    births_per_month = {}
    for idx in range(1,13):
        val_sum = 0
        for row in nested_list:
            if row[1] == idx:
                val_sum += row[4]
        births_per_month[idx] = val_sum
    return births_per_month
cdc_month_births = month_births(read_csv("births.csv"))
cdc_month_births


```




    {1: 3232517,
     2: 3018140,
     3: 3322069,
     4: 3185314,
     5: 3350907,
     6: 3296530,
     7: 3498783,
     8: 3525858,
     9: 3439698,
     10: 3378814,
     11: 3171647,
     12: 3301860}




```python
def dow_births(nested_list):
     """
    input:
        nested_list - list of list containing output birth data
        where each list consist of year, month, day of month
        day of week, and births for that day as elements
    Output:
        returns a dictionary of days of week as keys and total number
        of births for each unique day of the week for the entire
        dataset as values
    """
    births_per_week = {}
    for idx in range(1,8):
        val_sum = 0
        for row in nested_list:
            if row[3] == idx:
                val_sum += row[4]
        births_per_week[idx] = val_sum
    return births_per_week
cdc_day_births = dow_births(read_csv("births.csv"))
cdc_day_births



```




    {1: 5789166,
     2: 6446196,
     3: 6322855,
     4: 6288429,
     5: 6233657,
     6: 4562111,
     7: 4079723}



<p>3. Write generic function that can aggregate births based on specified column</p>


```python
def calc_counts(data, column):
    """
    input:
        data - list of list containing output birth data
        where each list consist of year, month, day of month
        day of week, and births for that day as elements

        column - the column number we want to calculate the
        totals for i.e the index of the dataset's header
    Output:
        Populates and returns a dictionary containing the total number
        of births for each unique value in the column at position column
    """
    total_counts = {}
    column_val = list(map(lambda a: a[column], data))
    for idx in range(min(column_val),max(column_val)+1):
        val_sum = 0
        for row in data:
            if row[column] == idx:
                val_sum += row[4]
        total_counts[idx] = val_sum
    return total_counts
cdc_year_births = calc_counts(read_csv("births.csv"), 0)
#cdc_year_births
# cdc_month_births = calc_counts(read_csv("US_births_1994-2003_CDC_NCHS.csv"), 1)
# cdc_dom_births = calc_counts(read_csv("US_births_1994-2003_CDC_NCHS.csv"), 2)
# cdc_dow_births = calc_counts(read_csv("US_births_1994-2003_CDC_NCHS.csv"), 3)

```


```python
def dict_min_max_values(adict):
    """
    input:
        adict - any non-empty dictionary
    ouput:
        Returns a tuple of min and max values for
        any dictionary that's passed in
    """
    return (min(adict.values()), max(adict.values()))
dict_min_max_values(cdc_year_births)
```




    (3880894, 4089950)



<p>4. Write a function that determines the year on year change in aggregate date based on a row.
With this, answer the question; "How did the number of aggregate births on saturdays change between 1994 and 2003?"</p>


```python
data_dict_column = {
    0: [1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003],
    1: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    2: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
    3: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
}


def year_on_year_change(nested_list,year_1, year_2, row_key):
    """
    Inputs:
        nested_list - csvfile read into a list of lists
        year_1 - start year of comparison
        year_2 - end year of comparison
        row_key - defines specific row values used to evaluate year-on-year
        change
    Output:
         the differences between consecutive values to show
         if number of births is increasing or decreasing.
    """
    refined_list = []
    for key, alist in data_dict_column.items():
        if row_key in alist:
            column_idx = key
            row_idx = alist.index(row_key) + 1
    for row in nested_list:
        if row[0] in (year_1, year_2) and row_idx == row[column_idx]:
            refined_list.append(row)
#     print(refined_list)
#     print(calc_counts(refined_list, 0)[1994])
    year_1_vals = calc_counts(refined_list, 0)[year_1]
    year_2_vals = calc_counts(refined_list, 0)[year_2]
    val_change = year_2_vals - year_1_vals
    header = ['year','month','date_of_month','day_of_week','births']
    print("The number of {} births between {} and {} changed by {}".format(row_key, year_1, year_2, val_change))

year_on_year_change(read_csv("births.csv"),1994, 2003, "Saturday")
```

    The number of Saturday births between 1994 and 2003 changed by -27287
