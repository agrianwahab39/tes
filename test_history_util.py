from utils import save_analysis_to_history, load_analysis_history
import os

# Define dummy data for testing
test_image_name = "test_image_01.jpg"
test_summary = {
    'type': 'Splicing Detected',
    'confidence': 'High',
    'copy_move_score': 20,
    'splicing_score': 85
}
test_processing_time = "12.34s"

history_file = 'analysis_history.json'

# Clean up any existing history file before test
if os.path.exists(history_file):
    os.remove(history_file)
    print(f"Removed existing {history_file} for fresh test.")

# Test 1: Save an entry
print(f"Attempting to save entry for {test_image_name}...")
save_analysis_to_history(test_image_name, test_summary, test_processing_time)
print("Save operation completed.")

# Verify file creation
if os.path.exists(history_file):
    print(f"{history_file} created successfully.")
else:
    print(f"ERROR: {history_file} was NOT created.")

# Test 2: Load history
print("Attempting to load history...")
history_data = load_analysis_history()
print(f"Load operation completed. Loaded {len(history_data)} entries.")

if history_data:
    print("History data content:")
    for entry in history_data:
        print(entry)

    # Validate content of the first entry (should be our test entry)
    if len(history_data) == 1:
        first_entry = history_data[0]
        if first_entry['image_name'] == test_image_name and \
           first_entry['analysis_summary']['type'] == test_summary['type'] and \
           first_entry['processing_time'] == test_processing_time:
            print("SUCCESS: First entry content matches test data.")
        else:
            print("ERROR: First entry content does NOT match test data.")
    else:
        print("ERROR: Expected 1 entry, found more or less.")

elif len(history_data) == 0 and os.path.exists(history_file):
     print(f"ERROR: {history_file} exists but no data was loaded. Check save/load logic or JSON format.")
else:
    print("ERROR: No history data loaded and file might not exist.")

# Test 3: Save another entry to test appending
test_image_name_2 = "test_image_02.png"
test_summary_2 = {
    'type': 'Copy-Move Detected',
    'confidence': 'Medium',
    'copy_move_score': 70,
    'splicing_score': 10
}
test_processing_time_2 = "8.76s"

print(f"Attempting to save second entry for {test_image_name_2}...")
save_analysis_to_history(test_image_name_2, test_summary_2, test_processing_time_2)
print("Second save operation completed.")

history_data_after_append = load_analysis_history()
print(f"Load operation after append completed. Loaded {len(history_data_after_append)} entries.")

if len(history_data_after_append) == 2:
    print("SUCCESS: History append seems to work. Found 2 entries.")
    second_entry = history_data_after_append[1] # newest is usually last if appended
    if second_entry['image_name'] == test_image_name_2 and \
       second_entry['analysis_summary']['type'] == test_summary_2['type']:
        print("SUCCESS: Second entry content matches test data.")
    else:
        print("ERROR: Second entry content does NOT match test data.")
else:
    print(f"ERROR: Expected 2 entries after append, found {len(history_data_after_append)}.")
