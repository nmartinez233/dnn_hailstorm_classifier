import tempfile
import pytz
from datetime import datetime
import pyart
import nexradaws

conn = nexradaws.NexradAwsInterface()

central_timezone = pytz.timezone('UTC')
radar_id = 'KLOT'
start = central_timezone.localize(datetime(2020,7,24,0,0))
end = central_timezone.localize (datetime(2020,7,25,0,0))
scans = conn.get_avail_scans_in_range(start, end, radar_id)
print("There are {} scans available between {} and {}\n".format(len(scans), start, end))
print(scans[0:4])

results = conn.download(scans, '../data/NEXRAD/')