-- 1. Top 10 countries by total confirmed cases.
select
Country,
confirmed_cases
from country_wise_latest_covid_data
order by confirmed_cases desc
limit 10;

-- 2. Total number of deaths worldwide.
select
sum(deaths) as total_deaths
from country_wise_latest_covid_data;

-- 3. Countries with the highest death rate (Deaths / 100 Cases).
select 
Country,
deaths_per_100cases
from country_wise_latest_covid_data
order by deaths_per_100cases desc
limit 10;

-- 4. Countries with the highest recovery rate (Recovered / 100 Cases).
select 
Country,
recovered_per_100cases
from country_wise_latest_covid_data
order by recovered_per_100cases desc
limit 10;

-- 5. Total confirmed, deaths, and recovered cases by WHO Region.
select 
who_region,
sum(confirmed_cases) as total_confirmed_cases,
sum(deaths) as total_deaths,
sum(recovered) as total_recoverd
from country_wise_latest_covid_data
group by who_region;

-- 6. Countries with the largest increase in confirmed cases over the last week.
select 
Country,
one_week_change
from country_wise_latest_covid_data
order by one_week_change desc
limit 10;

-- 7. Countries with more than 10,000 active cases.
select 
Country,
active_cases
from country_wise_latest_covid_data
where active_cases > 10000;

-- 8. Weekly growth percentage by WHO Region.
select 
who_region,
round(avg(one_week_percent_increase),2) as avg_growth_rate
from country_wise_latest_covid_data
group by who_region;

-- 9. Top 5 countries with the most new cases today.
select 
Country,
new_cases
from country_wise_latest_covid_data
order by new_cases desc
limit 5;

-- 10. Ratio of deaths to recoveries in each country.
select 
Country,
deaths_per_100recovered as death_recovery_ratio
from country_wise_latest_covid_data
order by death_recovery_ratio desc;

-- 11. Average death rate (Deaths / 100 Cases) across all countries.
select
avg(deaths_per_100cases) as avg_death_rate
from country_wise_latest_covid_data;

/* 12. Comparison of confirmed cases and recovered cases in the 
top 10 countries by confirmed cases.*/
select 
Country,
confirmed_cases,
recovered
from country_wise_latest_covid_data
order by confirmed_cases desc
limit 10;

-- 13. Cumulative death rate by country (top 10 countries).
select 
Country,
sum(deaths)/sum(confirmed_cases)*100 as death_rate
from country_wise_latest_covid_data
group by Country
order by death_rate desc
limit 10;

-- 14. Countries with the most significant decrease in new deaths.
select 
Country,
new_deaths
from country_wise_latest_covid_data
where new_deaths < 10
order by new_deaths asc;

/* 15. Countries with confirmed cases less than 
1 week ago but no new cases in the last 24 hours.*/
select 
Country,
confirmed_cases,
new_cases
from country_wise_latest_covid_data
where new_cases = 0 and confirmed_last_week > 0;