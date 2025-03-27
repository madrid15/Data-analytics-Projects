create database ipl;
use ipl;
select * from ipl_2023_batsman;
select * from ipl_2023_bowler; 
SELECT * FROM  ipl_2023_match_scoreboard;
SELECT * FROM ipl_2023_matches;

-- 1) Retrieve all batsmen who scored more than 50 runs in a single match.
SELECT DISTINCT
    Batsman, team, Run
FROM
    ipl_2023_batsman
WHERE
    run > 50;
-- 2)Find all bowlers who took 3 or more wickets in a single match.

SELECT 
    Bowler, team, wicket
FROM
    ipl_2023_bowler
WHERE
    wicket >= 3;

-- 3)List the matches played in Chennai. 
SELECT 
    *
FROM
    ipl_2023_matches
WHERE
    city = 'Chennai'; 

-- 4) Display the names of players who hit more than 4 sixes in a single match. 

SELECT DISTINCT
    Batsman
FROM
    ipl_2023_batsman
WHERE
    six > 4; 
-- 5) Retrieve the team with the highest total score in a match. 

SELECT match_no, 
       CASE 
           WHEN Home_team_run > Away_team_run THEN Home_team_run
           ELSE Away_team_run
       END AS Team,
       GREATEST(Home_team_run, Away_team_run) AS Total_Score
FROM ipl_2023_match_scoreboard
ORDER BY Total_Score DESC
LIMIT 1;

-- 6) List all matches where the home team lost.
SELECT 
    u.Home_team
FROM
    ipl_2023_matches u
        LEFT JOIN
    ipl_2023_match_scoreboard h ON u.match_no = h.match_no
WHERE
    h.Home_team_run < Away_team_run; 

-- 7)Retrieve the man of the match for all matches won by Chennai Super Kings.

SELECT 
    match_no, man_of_the_match, winner
FROM
    ipl_2023_matches
WHERE
    winner = 'Chennai Super Kings';


-- 8)List all matches that ended in a tie.  
SELECT 
    match_no
FROM
    ipl_2023_match_scoreboard
WHERE
    Home_team_run = Away_team_run; 

-- 9) Calculate the average score of each team in the tournament.

SELECT DISTINCT
    u.Home_team,
    AVG(v.Home_team_run) AS home_team_avg,
    AVG(v.Away_team_run) AS away_team_avg
FROM
    ipl_2023_match_scoreboard v
        INNER JOIN
    ipl_2023_matches u ON u.match_no = v.match_no
GROUP BY u.Home_team;

-- 10) Find the player who scored the maximum runs across the entire tournament.
SELECT 
    Batsman, sum(Run) AS max_run
FROM
    ipl_2023_batsman
GROUP BY Batsman
ORDER BY max_run DESC
LIMIT 1;

-- 11) Identify the bowler with the highest number of wickets in the season.
SELECT 
    Bowler, SUM(wicket) AS max_wicket
FROM
    ipl_2023_bowler
GROUP BY Bowler
ORDER BY max_wicket DESC
LIMIT 1;

-- 12) List the number of sixes hit by each batsman in the tournament.
SELECT 
    batsman, SUM(six) AS total_six
FROM
    ipl_2023_batsman
GROUP BY batsman
ORDER BY total_six DESC; 

 -- 13) Find the city that hosted the maximum number of matches.
 SELECT 
    COUNT(match_no) AS total_match, city
FROM
    ipl_2023_matches
GROUP BY city
ORDER BY total_match DESC;

-- 14) Calculate the strike rate for all batsmen.
SELECT DISTINCT
    Batsman, (SUM(Run) / SUM(Ball) * 100) AS strike_rate
FROM
    ipl_2023_batsman
GROUP BY Batsman
ORDER BY strike_rate DESC; 

-- 15)List all bowlers who bowled a no-ball in a match.
SELECT 
    Bowler
FROM
    ipl_2023_bowler
WHERE
    No_ball = 1;
-- 16) Find the match with the highest margin of victory.

SELECT 
    Home_team, Away_team, MAX(result_margin) AS high_margin
FROM
    ipl_2023_matches
GROUP BY Home_team , Away_team
ORDER BY high_margin DESC
LIMIT 1;


-- 17) Determine the team that won the most tosses during the season.
SELECT toss_winner, COUNT(*) AS toss_wins
FROM ipl_2023_matches
GROUP BY toss_winner
ORDER BY toss_wins DESC
LIMIT 1;

-- 18) Calculate the average number of wickets lost by the home team in all matches

SELECT 
    u.Home_team,
    SUM(v.Home_team_run) / sum(v.Home_team_wickets) AS avg_number_wicket_lost_by_home_team
FROM
    ipl_2023_matches u
        INNER JOIN
    ipl_2023_match_scoreboard v ON u.match_no = v.match_no
GROUP BY u.Home_team order by avg_number_wicket_lost_by_home_team desc; 

-- 19) Find the batsman with the highest strike rate in matches where their team won.
SELECT DISTINCT
    u.winner,
    v.Batsman,
    (SUM(v.Run) / SUM(v.Ball) * 100) AS strike_rate
FROM
    ipl_2023_batsman v
        INNER JOIN
    ipl_2023_matches u ON v.match_no = u.match_no
WHERE
    u.winner = u.Home_team
        OR u.winner = u.Away_team
GROUP BY u.winner , v.Batsman
ORDER BY strike_rate DESC;  

-- 20) Identify the most consistent batsman (highest average runs per match).

SELECT 
    Batsman,
    (SUM(Run) / COUNT(match_no)) AS highest_average_runs_per_match
FROM
    ipl_2023_batsman
GROUP BY Batsman
ORDER BY highest_average_runs_per_match DESC; 
-- 21) Find the match with the highest number of sixes hit across both teams.
SELECT 
    u.Home_team,
    u.Away_team,
    COUNT(v.six) AS highest_number_of_six
FROM
    ipl_2023_matches u
        INNER JOIN
    ipl_2023_batsman v ON u.match_no = v.match_no
GROUP BY u.Home_team , u.Away_team
ORDER BY highest_number_of_six DESC; 
-- 22) List the teams that won all their matches played as the home team.
SELECT 
    winner, Home_team
FROM
    ipl_2023_matches
WHERE
    winner = Home_team
;

-- 23) Find the umpire who officiated in the highest number of matches. 

SELECT 
    umpire_name, 
    COUNT(*) AS matches_officiated
FROM 
    (
        SELECT umpire1 AS umpire_name FROM ipl_2023_matches
        UNION ALL
        SELECT umpire2 AS umpire_name FROM ipl_2023_matches
    ) AS umpires
GROUP BY 
    umpire_name
ORDER BY 
    matches_officiated DESC; 
-- 24) Identify the venue where the highest number of runs were scored in a single match.
SELECT 
    distinct u.venue, v.Home_team_run + v.Away_team_run AS total_run
FROM
    ipl_2023_matches u
        INNER JOIN
    ipl_2023_match_scoreboard v ON u.match_no = v.match_no
ORDER BY total_run DESC limit 1;

-- 25) Find the total number of sixes hit in the tournament.
SELECT 
    SUM(six) AS total_num_six
FROM
    ipl_2023_batsman; 

-- 26) Determine the player who received the most man of the match awards. 
 SELECT 
    man_of_the_match, COUNT(*) AS most_award_received
FROM
    ipl_2023_matches
GROUP BY man_of_the_match
ORDER BY most_award_received DESC; 

-- 27) Calculate the total runs and wickets of each team in away matches. 
SELECT 
    SUM(u.Away_team_run) AS away_team_total_run,
    SUM(u.Home_team_wickets) AS away_team_total_wicket,
    v.Away_team
FROM
    ipl_2023_match_scoreboard u
        INNER JOIN
    ipl_2023_matches v ON u.match_no = v.match_no
GROUP BY v.Away_team; 

-- 28) Retrieve the batsman who hit the most fours in a single match
SELECT 
    Batsman, max(four) AS most_four
FROM
    ipl_2023_batsman
GROUP BY Batsman
;  

-- 29) Find the match with the maximum wickets taken by bowlers from a single team
 SELECT 
    Bowler, MAX(wicket) AS max_wicket_taken
FROM
    ipl_2023_bowler
GROUP BY Bowler
ORDER BY max_wicket_taken DESC; 

-- 30) Identify the most successful team (highest win percentage).
SELECT 
    team, 
    ROUND((wins / matches_played) * 100, 2) AS win_percentage
FROM (
    SELECT 
        team,
        SUM(CASE WHEN team = winner THEN 1 ELSE 0 END) AS wins,
        COUNT(*) AS matches_played
    FROM (
        SELECT Home_team AS team, winner FROM ipl_2023_matches
        UNION ALL
        SELECT Away_team AS team, winner FROM ipl_2023_matches
    ) AS all_matches
    GROUP BY team
) AS team_stats
ORDER BY win_percentage DESC
;

--  31)Find the player who scored the fastest fifty (minimum balls faced).

SELECT 
    Batsman, Run, MIN(Ball) AS minimum_ball
FROM
    ipl_2023_batsman
WHERE
    Run = 50
GROUP BY Run , Batsman
ORDER BY minimum_ball
LIMIT 1;




-- 32) Calculate the overall economy rate for each bowler in the tournament.

SELECT 
    Bowler, SUM(run) / SUM(over_) AS economy_rate
FROM
    ipl_2023_bowler
GROUP BY Bowler order by economy_rate 
;

-- 33)Calculate the percentage of matches won by teams batting first.
SELECT 
    ROUND((COUNT(*) * 100.0 / (SELECT COUNT(*) FROM ipl_2023_matches)), 2) AS batting_first_win_percentage
FROM ipl_2023_matches m
JOIN ipl_2023_match_scoreboard ms ON m.match_no = ms.match_no
WHERE 
    (m.Home_team = m.winner AND ms.Home_team_run > ms.Away_team_run)
    OR 
    (m.Away_team = m.winner AND ms.Away_team_run > ms.Home_team_run);

-- 34) Identify the team that defended the lowest total successfully.
SELECT 
    m.match_no,
    m.Home_team AS defending_team,
    ms.Home_team_run AS defended_score,
    m.Away_team AS chasing_team,
    ms.Away_team_run AS chasing_score
FROM 
    ipl_2023_matches m
JOIN 
    ipl_2023_match_scoreboard ms ON m.match_no = ms.match_no
WHERE 
    ms.Home_team_run > ms.Away_team_run
UNION ALL
SELECT 
    m.match_no,
    m.Away_team AS defending_team,
    ms.Away_team_run AS defended_score,
    m.Home_team AS chasing_team,
    ms.Home_team_run AS chasing_score
 
FROM 
    ipl_2023_matches m
JOIN 
    ipl_2023_match_scoreboard ms ON m.match_no = ms.match_no
WHERE 
    ms.Away_team_run > ms.Home_team_run
ORDER BY 
    defended_score ASC
limit 1;

-- 35) Determine the most consistent bowler based on wickets per match.

SELECT 
    Bowler,
    SUM(wicket) / COUNT(DISTINCT (match_no)) AS wicket_per_match
FROM
    ipl_2023_bowler
GROUP BY Bowler order by wicket_per_match desc ; 