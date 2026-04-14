Python app → log file → Promtail → Loki → Grafana

1. Architecture
Loki = stores logs
Promtail = reads your log file
Grafana = dashboards
2. Put logs in folder
Put app.log in  logs

3. Start docker container
docker-compose up
4. Open Grafana
http://localhost:3000
admin / admin
5. Connect Loki
Go to Connections → Data Sources
Add Loki
URL: http://loki:3100
6. Query logs
Go to Explore and run:
{job="my_app"}
7. Parse your JSON logs:
{job="my_app"} | json

8. Build metrics from logs
Throughput (logs per second)
count_over_time({job="my_app"}[1m])
Error rate
count_over_time({job="my_app"} | json | message_status="Error" [1m])
Average duration
avg_over_time(
  {job="my_app"} 
  | json 
  | unwrap message_duration 
  [1m]
)

p95 latency
quantile_over_time(
  0.95,
  {job="my_app"} 
  | json 
  | unwrap message_duration 
  [5m]
)
Group by operation type
sum by (message_operation_type) (
  count_over_time({job="my_app"} | json [1m])
)

9. Turn into dashboards

In Grafana:

Click + → Dashboard
Add panel
Paste one of the queries above
Choose:
Time series (for latency)
Stat (for totals)
Bar chart (for breakdowns)

10. Test it

Append logs manually:
echo '{"message":{"duration":2,"status":"Success"}}' >> logs/app.log
Watch Grafana update in real time.