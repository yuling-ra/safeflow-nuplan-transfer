| Bucket        | Method            | Training   |   Safety |     Goal |   TimeToGoal |   MinDist |   RoadViol |   N |
|:--------------|:------------------|:-----------|---------:|---------:|-------------:|----------:|-----------:|----:|
| 2car_crossing | FM+MPPI           | zero_shot  | 1        | 0.411765 |      13.8647 |  0.683727 |          0 |  17 |
| 2car_crossing | FM+MPPI+CBF(Exec) | zero_shot  | 1        | 0        |      15.1    |  2.15888  |          0 |  17 |
| 2car_crossing | SafeFlow (FM+CBF) | zero_shot  | 1        | 0.647059 |      13.2471 |  0.813586 |          0 |  17 |
| 2car_headon   | FM+MPPI           | zero_shot  | 0.941176 | 0.529412 |      13.9353 |  0.494691 |          0 |  17 |
| 2car_headon   | FM+MPPI+CBF(Exec) | zero_shot  | 1        | 0        |      15.1    |  1.75705  |          0 |  17 |
| 2car_headon   | SafeFlow (FM+CBF) | zero_shot  | 1        | 0.529412 |      13.7706 |  0.486906 |          0 |  17 |
| 2car_parallel | FM+MPPI           | zero_shot  | 0.9375   | 0.5      |      13.9313 |  0.324533 |          0 |  16 |
| 2car_parallel | FM+MPPI+CBF(Exec) | zero_shot  | 1        | 0        |      15.1    |  0.365952 |          0 |  16 |
| 2car_parallel | SafeFlow (FM+CBF) | zero_shot  | 0.8125   | 0.625    |      13.275  |  0.288215 |          0 |  16 |