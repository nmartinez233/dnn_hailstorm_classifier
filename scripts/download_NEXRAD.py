import tempfile
import pytz
from datetime import datetime
import pyart
import nexradaws

conn = nexradaws.NexradAwsInterface()

central_timezone = pytz.timezone('UTC')
radar_id = 'KLZK'
start = central_timezone.localize(datetime(2020,5,1,0,0))
end = central_timezone.localize (datetime(2020,8,16,0,0))
scans = conn.get_avail_scans_in_range(start, end, radar_id)
print("There are {} scans available between {} and {}\n".format(len(scans), start, end))


results = conn.download(scans, '../data/KLZK_data')

#print(results.success)
#print(results.failed)
