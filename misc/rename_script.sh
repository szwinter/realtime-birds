#!/bin/bash

# Script to rename files from a pattern like 'ANYTHING_db_data_YYYY_M_D.csv'
# to 'metadata_YYYY_MM_DD.csv'.
# YYYY, M, D are year, month, day. MM and DD are zero-padded month and day.

# --- Configuration ---
TARGET_PREFIX="observations"

echo "Shell script to rename files"
echo "Original pattern: ANYTHING_db_data_YEAR_MONTH_DAY.csv"
echo "Target pattern:   ${TARGET_PREFIX}_YEAR_PADDEDMONTH_PADDEDDAY.csv"
echo "------------------------------------------------------------------"
echo

# Use shopt -s nullglob to make the glob expand to nothing if no files match
shopt -s nullglob
# The glob looks for files containing "_db_data_" followed by at least one underscore (for year_month_day)
# and ending in .csv. It assumes "ANYTHING" part can contain underscores too.
files_to_process=(*_db_data_*_*_*.csv)
shopt -u nullglob # Reset nullglob to its default behavior

if [ ${#files_to_process[@]} -eq 0 ]; then
    echo "No files found matching a pattern like '*_db_data_Y_M_D.csv' in the current directory."
    exit 0
fi

echo "Proposed renames:"
echo "=================================================================="

# Associative array to store old_name -> new_name mappings
declare -A rename_map

for old_filename in "${files_to_process[@]}"; do
    # Extract the part after the last occurrence of "_db_data_"
    # e.g., for "00000_db_data_2023_2_1.csv", date_part_with_csv becomes "2023_2_1.csv"
    # Using ##* to ensure we get the part after the *last* instance of _db_data_
    date_part_with_csv="${old_filename##*_db_data_}"

    # Remove the .csv extension
    # e.g., "2023_2_1.csv" becomes "2023_2_1"
    date_part="${date_part_with_csv%.csv}"

    # Split the date_part by underscore.
    # The 'IFS= '_' prefix limits the scope of IFS change to the read command.
    # -r prevents backslash escapes from being processed.
    # We expect 3 parts: year, month, day. extra_parts should be empty.
    IFS='_' read -r year month day extra_parts <<< "$date_part"

    # Validate extracted components:
    # 1. Year, Month, Day must be non-empty.
    # 2. Year, Month, Day must be numeric.
    # 3. There should be no extra parts after day (ensures Y_M_D structure).
    if ! [[ -n "$year" && "$year" =~ ^[0-9]+$ && \
            -n "$month" && "$month" =~ ^[0-9]+$ && \
            -n "$day" && "$day" =~ ^[0-9]+$ && \
            -z "$extra_parts" ]]; then
        echo "Skipping '$old_filename': Could not parse a clear YEAR_MONTH_DAY structure from name segment '$date_part'."
        continue
    fi

    # Basic validation for month and day ranges.
    # In Bash arithmetic ((...)), numbers with leading zeros can be treated as octal.
    # Using 10# prefix ensures decimal interpretation (e.g., 10#08 is 8, 10#09 is 9).
    if ! (( 10#$month >= 1 && 10#$month <= 12 && 10#$day >= 1 && 10#$day <= 31 )); then
        echo "Skipping '$old_filename': Month ($month) or Day ($day) out of valid range (Month: 1-12, Day: 1-31)."
        continue
    fi

    # Format month and day to be two digits (e.g., 2 -> 02, 10 -> 10)
    # printf handles "08" or "8" correctly for %d.
    formatted_month=$(printf "%02d" "$month")
    formatted_day=$(printf "%02d" "$day")

    # Construct the new filename
    new_filename="${TARGET_PREFIX}_${year}_${formatted_month}_${formatted_day}.csv"

    # Check if new name is actually different from the old name
    if [ "$old_filename" == "$new_filename" ]; then
        echo "Skipping '$old_filename': New name would be identical."
        continue
    fi
    
    # Store for later processing
    rename_map["$old_filename"]="$new_filename"
    
    # Print proposed change, aligning columns with printf
    printf "%-45s => %s\n" "'$old_filename'" "'$new_filename'"

done

echo "=================================================================="

if [ ${#rename_map[@]} -eq 0 ]; then
    echo "No files eligible for renaming after validation and checks."
    exit 0
fi

echo
read -p "Proceed with the ${#rename_map[@]} rename(s) listed above? (Type 'yes' to confirm): " confirmation

if [[ "$confirmation" == "yes" || "$confirmation" == "YES" ]]; then
    echo
    echo "Proceeding with renaming..."
    success_count=0
    fail_count=0
    skipped_due_to_existing_target=0

    for old_filename in "${!rename_map[@]}"; do
        new_filename="${rename_map[$old_filename]}"

        # Check if target file already exists before attempting to move
        if [ -e "$new_filename" ]; then
            # You can choose to use 'mv -i' for interactive, or 'mv -n' for no-clobber.
            # If using 'mv -n', the file will be skipped.
            # If using 'mv -i', user will be prompted.
            # Here, we use 'mv -i' to give user control over overwrites.
            echo -n "Target '$new_filename' exists. "
            if mv -i "$old_filename" "$new_filename"; then
                echo "Renamed '$old_filename' to '$new_filename' (user confirmed overwrite or target did not block)."
                ((success_count++))
            else
                echo "Skipped renaming '$old_filename' to '$new_filename' (mv command failed or user declined overwrite)."
                ((fail_count++))
            fi
        else
            if mv "$old_filename" "$new_filename"; then
                echo "Renamed '$old_filename' to '$new_filename'"
                ((success_count++))
            else
                echo "Error renaming '$old_filename' to '$new_filename'."
                ((fail_count++))
            fi
        fi
    done
    echo
    echo "Renaming process complete."
    echo "Successfully renamed: $success_count file(s)."
    echo "Failed or skipped: $fail_count file(s)."
else
    echo "Renaming aborted by user."
fi

exit 0
