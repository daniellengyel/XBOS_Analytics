from Occupancy import Occupancy
import pandas as pd

from flask import Flask, request, jsonify, url_for
app = Flask(__name__, static_url_path="")

Occ = Occupancy()

@app.route('/test', methods=['GET', 'POST'])
def test():
	print(request.method)
	assert request.method == 'POST'
	day = request.form["day"]
	print(day)
	day = pd.to_datetime(day, format='%Y%m%d', errors='ignore')
	numClasses = request.form["numSameClasses"]
	print(numClasses)
	numDays = request.form['numSameDays']
	print(numDays)
	cutoffPercentage = request.form['cutoffPercentage']
	print(cutoffPercentage)
	building = request.form.get('building')
	print(building)
	zone = request.form.get('zone')
	print(zone)
	if "Zone" in zone:
		schedule = Occ.adaptive_schedule(Occ.zone_df[zone], day, numClasses, numDays)
	else:
		schedule = Occ.adaptive_schedule(Occ.building_df, day, numClasses, numDays)
	return schedule.to_json()


@app.route('/')
def root():
	return app.send_static_file('./WebApp.html')
