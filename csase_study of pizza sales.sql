create database pizza_hurt;
use pizza_hurt;
CREATE TABLE runners (
  runner_id INTEGER,
  registration_date DATE
);
INSERT INTO runners
  (runner_id, registration_date)
VALUES
  (1, '2021-01-01'),
  (2, '2021-01-03'),
  (3, '2021-01-08'),
  (4, '2021-01-15');
  select * from runners; 
CREATE TABLE customer_orders (
  order_id INTEGER,
  customer_id INTEGER,
  pizza_id INTEGER,
  exclusions VARCHAR(4),
  extras VARCHAR(4),
  order_time TIMESTAMP
);
INSERT INTO customer_orders
  (order_id, customer_id, pizza_id, exclusions, extras, order_time)
VALUES
  ('1', '101', '1', '', '', '2020-01-01 18:05:02'),
  ('2', '101', '1', '', '', '2020-01-01 19:00:52'),
  ('3', '102', '1', '', '', '2020-01-02 23:51:23'),
  ('3', '102', '2', '', NULL, '2020-01-02 23:51:23'),
  ('4', '103', '1', '4', '', '2020-01-04 13:23:46'),
  ('4', '103', '1', '4', '', '2020-01-04 13:23:46'),
  ('4', '103', '2', '4', '', '2020-01-04 13:23:46'),
  ('5', '104', '1', 'null', '1', '2020-01-08 21:00:29'),
  ('6', '101', '2', 'null', 'null', '2020-01-08 21:03:13'),
  ('7', '105', '2', 'null', '1', '2020-01-08 21:20:29'),
  ('8', '102', '1', 'null', 'null', '2020-01-09 23:54:33'),
  ('9', '103', '1', '4', '1, 5', '2020-01-10 11:22:59'),
  ('10', '104', '1', 'null', 'null', '2020-01-11 18:34:49'),
  ('10', '104', '1', '2, 6', '1, 4', '2020-01-11 18:34:49');
select * from customer_orders;

CREATE TABLE runner_orders (
  order_id INTEGER,
  runner_id INTEGER,
  pickup_time  VARCHAR(19),
  distance VARCHAR(7),
  duration VARCHAR(10),
  cancellation VARCHAR(23)
);
INSERT INTO runner_orders
  (order_id, runner_id, pickup_time, distance, duration, cancellation)
VALUES
  ('1', '1', '2020-01-01 18:15:34', '20km', '32 minutes', ''),
  ('2', '1', '2020-01-01 19:10:54', '20km', '27 minutes', ''),
  ('3', '1', '2020-01-03 00:12:37', '13.4km', '20 mins', NULL),
  ('4', '2', '2020-01-04 13:53:03', '23.4', '40', NULL),
  ('5', '3', '2020-01-08 21:10:57', '10', '15', NULL),
  ('6', '3', 'null', 'null', 'null', 'Restaurant Cancellation'),
  ('7', '2', '2020-01-08 21:30:45', '25km', '25mins', 'null'),
  ('8', '2', '2020-01-10 00:15:02', '23.4 km', '15 minute', 'null'),
  ('9', '2', 'null', 'null', 'null', 'Customer Cancellation'),
  ('10', '1', '2020-01-11 18:50:20', '10km', '10minutes', 'null');
  select * from runner_orders;
  
  CREATE TABLE pizza_names (
  pizza_id INTEGER,
  pizza_name TEXT
);
INSERT INTO pizza_names
  (pizza_id, pizza_name)
VALUES
  (1, 'Meatlovers'),
  (2, 'Vegetarian');
 select * from pizza_names;
 
CREATE TABLE pizza_recipes (
  pizza_id INTEGER,
  toppings TEXT
);
INSERT INTO pizza_recipes
  (pizza_id, toppings)
VALUES
  (1, '1, 2, 3, 4, 5, 6, 8, 10'),
  (2, '4, 6, 7, 9, 11, 12');
  select * from pizza_recipes;
  
CREATE TABLE pizza_toppings (
  topping_id INTEGER,
  topping_name TEXT
);
INSERT INTO pizza_toppings
  (topping_id, topping_name)
VALUES
  (1, 'Bacon'),
  (2, 'BBQ Sauce'),
  (3, 'Beef'),
  (4, 'Cheese'),
  (5, 'Chicken'),
  (6, 'Mushrooms'),
  (7, 'Onions'),
  (8, 'Pepperoni'),
  (9, 'Peppers'),
  (10, 'Salami'),
  (11, 'Tomatoes'),
  (12, 'Tomato Sauce'); 
select * from pizza_toppings;

-- 1.How many pizzas were ordered?
SELECT 
    COUNT(order_id) AS total_pizzas_order
FROM
    customer_orders;

-- 2.How many unique customer orders were made?
SELECT 
    COUNT(DISTINCT customer_id) as total_customer_order
FROM
    customer_orders
;
-- 3. How many successful orders were delivered by each runner?
SELECT 
  COUNT(cancellation)
FROM
    runner_orders
WHERE
    distance not in( 'null' and NULL)  ; 

-- 4.How many of each type of pizza was delivered?
SELECT 
    p.pizza_name,
    COUNT(co.order_id) AS total_delivered
FROM 
    pizza_names p
    INNER JOIN customer_orders co ON p.pizza_id = co.pizza_id
GROUP BY 
    p.pizza_name;


-- 5.How many Vegetarian and Meatlovers were ordered by each customer?
SELECT 
 u.pizza_name,COUNT(u.pizza_name) AS diff_type_of_pizza
FROM
    pizza_names u
        INNER JOIN
    customer_orders v ON u.pizza_id = v.pizza_id
group by u.pizza_name;

-- 6.What was the maximum number of pizzas delivered in a single order?
SELECT DISTINCT
    order_id, COUNT(*) AS max_time_order
FROM
    customer_orders
GROUP BY order_id
ORDER BY max_time_order DESC
LIMIT 1;
-- 7. For each customer, how many delivered pizzas had at least 1 change and how many had no changes?
SELECT 
    customer_id,
    SUM(CASE 
            WHEN (exclusions IS NOT NULL AND exclusions != '') 
              OR (extras IS NOT NULL AND extras != '') THEN 1 
            ELSE 0 
        END) AS pizzas_with_changes,
    SUM(CASE 
            WHEN (exclusions IS NULL OR exclusions = '') 
              AND (extras IS NULL OR extras = '') THEN 1 
            ELSE 0 
        END) AS pizzas_without_changes
FROM 
    customer_orders
GROUP BY 
    customer_id;
-- 8.How many pizzas were delivered that had both exclusions and extras? 

SELECT 
    COUNT(*) AS total_pizza
FROM
    customer_orders
WHERE
    (exclusions IS NOT NULL
        AND exclusions <> ' ')
        AND (extras IS NOT NULL AND extras <> ' ');


-- 9. What was the total volume of pizzas ordered for each hour of the day?
SELECT 
    EXTRACT(HOUR FROM order_time) as_each_hour_order,
    COUNT(order_id) AS total_order
FROM
    customer_orders
GROUP BY EXTRACT(HOUR FROM order_time)
ORDER By as_each_hour_order; 

-- 10. What was the volume of orders for each day of the week?
SELECT 
    dayname(order_time) AS day_of_week,
    COUNT(order_id) AS volume_of_orders
FROM
    customer_orders
GROUP BY day_of_week
ORDER BY FIELD(day_of_week, 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday');  

-- 11.How many runners signed up for each 1 week period? (i.e. week starts 2021-01-01)
SELECT 
    COUNT(runner_id) AS total_runner,
    YEARWEEK(pickup_time, 1) AS week_period
FROM
    runner_orders
    where pickup_time>=2021-01-01
GROUP BY week_period
ORDER BY week_period;

-- 12. What was the average time in minutes it took for each runner to arrive at the Pizza Runner HQ to pickup the order? 
SELECT 
    distinct runner_id, AVG(MINUTE(pickup_time)) AS avg_time_taken
FROM
    runner_orders where pickup_time <> "null"
GROUP BY runner_id;

-- 13. Is there any relationship between the number of pizzas and how long the order takes to prepare?
SELECT 
    COUNT(u.order_id) AS num_of_pizza, v.duration
FROM
    customer_orders u
        INNER JOIN
    runner_orders v ON u.order_id = v.order_id
WHERE
    v.duration IS NOT NULL
        AND v.duration <> 'null'
GROUP BY v.duration
ORDER BY v.duration DESC limit 1 ;

-- 14.What was the average distance travelled for each customer?
SELECT 
    u.customer_id, ROUND(AVG(CAST(REPLACE(v.distance, 'km', '') AS DECIMAL(10, 2))), 2) AS avg_distance
FROM
    customer_orders u
        INNER JOIN
    runner_orders v ON u.order_id = v.order_id
WHERE
    v.distance IS NOT NULL
        AND v.distance <> 'null'
GROUP BY u.customer_id;

-- 15. What was the difference between the longest and shortest delivery times for all orders?

SELECT 
    MAX(CAST(REPLACE(Duration, 'minutes', '') AS UNSIGNED)) - MIN(CAST(REPLACE(Duration, 'minutes', '') AS UNSIGNED)) AS diff
FROM
    runner_orders
WHERE
    pickup_time IS NOT NULL
        AND pickup_time <> 'null'; 

-- 16. What was the average speed for each runner for each delivery and do you notice any trend for these values
SELECT 
    AVG(CAST(REPLACE(distance, 'km', '') AS DECIMAL(10, 2)) / 
        (CAST(REPLACE(duration, 'minutes', '') AS DECIMAL(10, 2)) / 60)) AS avg_speed_kmh
FROM
    runner_orders
WHERE
    distance IS NOT NULL AND distance <> 'null'
    AND duration IS NOT NULL AND duration <> 'null';

-- 17. What is the successful delivery percentage for each runner?
SELECT 
    r.runner_id,
    ROUND((SUM(CASE
                WHEN
                    ro.cancellation IS NULL
                        OR ro.cancellation = ''
                THEN
                    1
                ELSE 0
            END) / COUNT(*)) * 100,
            2) AS delivary_parcentage
FROM
    runners r
        LEFT JOIN
    runner_orders ro ON r.runner_id = ro.runner_id
GROUP BY r.runner_id;

-- 18. What are the standard ingredients for each pizza?
SELECT 
    pn.pizza_name AS pizza,
    GROUP_CONCAT(pt.topping_name ORDER BY pt.topping_id SEPARATOR ', ') AS standard_ingredients
FROM 
    pizza_recipes pr
JOIN 
    pizza_names pn ON pr.pizza_id = pn.pizza_id
JOIN 
    pizza_toppings pt ON FIND_IN_SET(pt.topping_id, pr.toppings)
GROUP BY 
    pn.pizza_id, pn.pizza_name;

-- 19. What was the most commonly added extra?
SELECT 
    pt.topping_name AS most_common_extra,
    COUNT(pt.topping_id) AS frequency
FROM 
    customer_orders co
JOIN 
    pizza_toppings pt ON FIND_IN_SET(pt.topping_id, co.extras)
WHERE 
    co.extras IS NOT NULL AND co.extras <> ''
GROUP BY 
    pt.topping_id, pt.topping_name
ORDER BY 
    frequency DESC
;

-- 20.What was the most common exclusion? 
SELECT 
    u.topping_name AS exclusion,
    COUNT(u.topping_id) AS frequency
FROM
    customer_orders v
        JOIN
    pizza_toppings u ON FIND_IN_SET(u.topping_id, v.exclusions)
WHERE
    v.exclusions IS NOT NULL
        AND v.exclusions <> 'null'
GROUP BY exclusion , u.topping_name
ORDER BY frequency DESC limit 1; 

-- 21. What is the total quantity of each ingredient used in all delivered pizzas sorted by most frequent first? 
SELECT 
    u.topping_name, COUNT(topping_id) AS total_quantity
FROM
    customer_orders w
        JOIN
    runner_orders x ON w.order_id = x.order_id
        JOIN
    pizza_recipes y ON w.pizza_id = y.pizza_id
        JOIN
    pizza_toppings u ON FIND_IN_SET(u.topping_id, y.toppings)
WHERE
    x.distance IS NOT NULL
GROUP BY u.topping_name
ORDER BY total_quantity DESC; 

-- 22. If a Meat Lovers pizza costs $12 and Vegetarian costs $10 and there were no charges for changes  
-- how much money has Pizza Runner made so far if there are no delivery fees?
 set @meat_cost=(select  count(u.pizza_name)*12 from customer_orders v join pizza_names u on u.pizza_id=v.pizza_id join runner_orders w on v.order_id=w.order_id where w.distance is not null and w.distance <>"null"  group by u.pizza_name order by count(u.pizza_name) desc limit 1);
 select  @meat_cost;
 set @vegeterian_cost= (select  count(u.pizza_name)*10 from customer_orders v join pizza_names u on u.pizza_id=v.pizza_id join runner_orders w on v.order_id=w.order_id where w.distance is not null and w.distance <>"null" group by u.pizza_name order by count(u.pizza_name)  limit 1);
 select @vegeterian_cost;
 select @meat_cost+@vegeterian_cost as total_revenue;
 

-- 23. What if there was an additional $1 charge for any pizza extras?

SELECT 
    SUM(CASE
        WHEN u.pizza_name = 'Meatlovers' THEN 12
        WHEN u.pizza_name = 'Vegetarian' THEN 10
        ELSE 0
    END) + COUNT(DISTINCT (y.topping_name)) AS total_revenue
FROM
    customer_orders w
        JOIN
    pizza_names u ON u.pizza_id = w.pizza_id
        JOIN
    runner_orders x ON w.order_id = x.order_id
        JOIN
    pizza_toppings y ON FIND_IN_SET(y.topping_id, w.extras)
WHERE
    w.extras IS NOT NULL
        AND w.extras <> 'null'
        AND x.distance IS NOT NULL
        AND x.distance <> 'null';

-- 24.Add cheese is $1 extra


SELECT 
    SUM(CASE
        WHEN u.pizza_name = 'Meatlovers' THEN 12
        WHEN u.pizza_name = 'Vegetarian' THEN 10
        ELSE 0
    END) + COUNT(DISTINCT (y.topping_name)) + SUM(CASE
        WHEN FIND_IN_SET('cheese', w.extras) > 0 THEN 1
        ELSE 0
    END) AS total_revenue
FROM
    customer_orders w
        JOIN
    pizza_names u ON u.pizza_id = w.pizza_id
        JOIN
    runner_orders x ON w.order_id = x.order_id
        JOIN
    pizza_toppings y ON FIND_IN_SET(y.topping_id, w.extras)
WHERE
    w.extras IS NOT NULL
        AND w.extras <> 'null'
        AND x.distance IS NOT NULL
        AND x.distance <> 'null';

-- 25.If a Meat Lovers pizza was $12 and Vegetarian $10 fixed prices with no cost for extras and each runner is paid $0.30 per kilometre traveled 
-- how much money does Pizza Runner have left over after these deliveries?

SELECT 
    SUM(CASE
        WHEN u.pizza_name = 'Meatlovers' THEN 12
        WHEN u.pizza_name = 'Vegetarian' THEN 10
        ELSE 0
    END) - SUM(CAST(REPLACE(v.duration, 'km', ' ') AS DECIMAL) * 0.30) AS total_revenue
FROM
    customer_orders w
        JOIN
    pizza_names u ON u.pizza_id = w.pizza_id
        JOIN
    runner_orders v ON w.order_id = v.order_id
WHERE
    v.distance IS NOT NULL
        AND v.distance <> 'null';