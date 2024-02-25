
#!/bin/bash

# * * * * * cd /Users/simen.eide@finn.no/Sync/weather/flysogn && ./cronjob.sh >> /Users/simen.eide@finn.no/Sync/weather/flysogn/cron.log

# print date and time
echo "Date: $(date) \t Running cronjob"
/Users/simen.eide@finn.no/mambaforge/envs/schib-lm/bin/python push_to_windy.py