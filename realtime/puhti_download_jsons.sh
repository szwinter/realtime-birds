#!/bin/bash
#SBATCH --job-name=download_jsons
#SBATCH --account=project_2003104
#SBATCH --output=output/%A
#SBATCH --ntasks=1 --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=22:15:00 --partition=small

REPLICATE=${1:-0}
COUNT=${2:-0}
COUNTMAX=${3:-100}
TARGET_TIME="${4:-03:00:00}"
echo $TARGET_TIME

# Validate the time format roughly (optional, but good practice)
# This regex checks for HH:MM:SS where HH is 00-23, MM is 00-59, SS is 00-59
if ! [[ "$TARGET_TIME" =~ ^([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$ ]]; then
  echo "Error: Invalid time format. Please use HH:MM:SS."
  exit 1
fi

# Get current timestamp in seconds since epoch
NOW_SECONDS=$(date +%s)

# Get today's date in YYYY-MM-DD format
TODAY_DATE=$(date +%F) # Equivalent to %Y-%m-%d

# Attempt to get the timestamp for the target time today
# Use '2>/dev/null' to suppress error messages from date if the time is invalid for today
# (though the regex check above should catch most issues)
TARGET_DATETIME_TODAY="${TODAY_DATE} ${TARGET_TIME}"
TARGET_TODAY_SECONDS=$(date -d "${TARGET_DATETIME_TODAY}" +%s 2>/dev/null)

# Check if date command was successful
if [ $? -ne 0 ]; then
  echo "Error: Could not parse the date/time: '${TARGET_DATETIME_TODAY}'"
  echo "Ensure the time '$TARGET_TIME' is valid."
  exit 1
fi

# Compare current time with the target time today
if [ "$NOW_SECONDS" -lt "$TARGET_TODAY_SECONDS" ]; then
  # Target time is later today
  RESULT_DATETIME_STR=$(date -d "${TARGET_DATETIME_TODAY}" +"%Y-%m-%d %H:%M:%S %Z")
  echo "The next occurrence of ${TARGET_TIME} is: ${RESULT_DATETIME_STR} (Today)"
else
  # Target time has already passed today, so it's tomorrow
  TOMORROW_DATE=$(date -d "tomorrow" +%F)
  TARGET_DATETIME_TOMORROW="${TOMORROW_DATE} ${TARGET_TIME}"
  RESULT_DATETIME_STR=$(date -d "${TARGET_DATETIME_TOMORROW}" +"%Y-%m-%d %H:%M:%S %Z")
  echo "The next occurrence of ${TARGET_TIME} is: ${RESULT_DATETIME_STR} (Tomorrow)"
fi


mkdir -p output
module load geoconda
hostname

srun python3 download_jsons.py

if (( COUNT < COUNTMAX )); then
	startTime=$(date -d "${RESULT_DATETIME_STR} +$COUNT days" +%Y-%m-%dT%H:%M:%S)
	COUNT=$((COUNT+1))
	echo $startTime
	sbatch --begin=$startTime puhti_download_jsons.sh $REPLICATE $COUNT $COUNTMAX $TARGET_TIME
fi
