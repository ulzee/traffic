
import time
import gmplot
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from glob import glob
import cv2
import json
import numpy as np

def get_manh_lines():
	polylogs = glob('../data/scraped/*.log')
	assert len(polylogs)

	all_polys = []
	for fname in polylogs:
		with open(fname) as fl:
			lines = fl.read().split('\n')


		polys = []
	#     bat = []
		bat = None
		route = None
		for ii, ln in enumerate(lines):
			if 'START' in ln:
				bat = []
				if route is None:
					route = ln.split()[-1]
				else:
					# check no log files were concatenated
					assert ln.split()[-1] == route
			elif 'END' in ln:
				polys.append(bat)
				bat = None
			elif bat is not None:
				lat, lng = ln.split(':')[1].split()[1:]
				bat.append((float(lat), float(lng)))
		# print(route, len(lines), len(polys))
		all_polys.append(polys)

	return all_polys

def map_manh(
	name, center=(40.78175269, -73.96522605),
	lines=None, colors=None, zoom=13, show=False, crop=60):

	man_lines = get_manh_lines()

	clan, clng = center

	gmap = gmplot.GoogleMapPlotter(
		clan,
		clng,
		zoom, 'AIzaSyASb7nA0mSFauWEbOmNOMdx10XDl_EFzVY')

	# scatter method of map object
	# scatter points on the google map
	# gmap.scatter( latitude_list, longitude_list, '# FF0000',
	# 							size = 40, marker = False )

	# Plot method Draw a line in
	# between given coordinates
	# gmap.plot(latitude_list, longitude_list,
	# 		'cornflowerblue', edge_width = 2.5)
	for polys in man_lines:
		for group in polys:
			lats, lngs = zip(*group)
			gmap.plot(lats, lngs, 'cornflowerblue', edge_width=2)

	gmap.draw( '.temp.html' )


	chrome_options = Options()
	chrome_options.add_argument("--headless")
	chrome_options.add_argument("--no-sandbox")

	driver = webdriver.Chrome(
		options = chrome_options)

	driver.get('file:///home/ubuntu/traffic/eval/.temp.html')
	time.sleep(1.5)
	driver.save_screenshot('%s.png' % name)
	driver.quit()

	img = cv2.cvtColor(cv2.imread('%s.png' % name), cv2.COLOR_BGR2RGB)
	cropped = img[crop:-crop, crop:-crop]
	cv2.imwrite('%s.png' % name, cropped)

	if show:
		import matplotlib.pyplot as plt
		plt.figure(figsize=(8, 7))
		plt.imshow(cropped)
		plt.show(); plt.close()

def map_graph(
	name, vs, adj,
	lines=None, colors=None, zoom=12, show=False, crop=60,
	wait=1.5,
	coords_file='/home/ubuntu/traffic/data/stop_coords.json'):

	with open(coords_file) as fl:
		coords = json.load(fl)

	stops = {}
	lines = []
	for seg in vs:
		verts = seg.split('-')
		line = []
		for stop in verts:
			if stop in coords:
				stops[stop] = coords[stop]
				line.append(coords[stop])
		if len(line) == 2:
			lines.append(np.array(line))
	gcoords = np.array(list(stops.values()))

	overlay = np.zeros((256, 256, 4)).astype(np.uint8)
	overlay[:,...,:3] = 255
	overlay[:,...,-1] = 125
	cv2.imwrite('overlay.png', overlay)

	clan, clng = np.mean(gcoords, axis=0)

	with open('../.keys.txt') as fl:
		gmkey = fl.read()

	gmap = gmplot.GoogleMapPlotter(
		clan,
		clng,
		zoom, gmkey)

	# bounds_dict = {'north':37.832285, 'south': 37.637336, 'west': -122.520364, 'east': -122.346922}
	# gmap.ground_overlay('file:///home/ubuntu/traffic/eval/overlay.png', bounds_dict)
	# assert len(gmap.ground_overlays)
	slats, slngs = zip(*gcoords)
	gmap.scatter(
		slats, slngs,
		'cornflowerblue',
		size=30, marker=False, face_alpha=1)

	for line in lines:
		lats, lngs = zip(*line)
		gmap.plot(lats, lngs, '#0099EE', edge_width=3)

	gmap.draw( 'temp.html' )


	chrome_options = Options()
	chrome_options.add_argument("--headless")
	chrome_options.add_argument("--no-sandbox")

	driver = webdriver.Chrome(
		options = chrome_options)


	driver.get('file:///home/ubuntu/traffic/eval/temp.html')
	time.sleep(wait)
	driver.save_screenshot('%s.png' % name)
	driver.quit()

	img = cv2.cvtColor(cv2.imread('%s.png' % name), cv2.COLOR_BGR2RGB)
	cropped = img[crop:-crop, crop:-crop]
	cv2.imwrite('%s.png' % name, cropped)

	if show:
		import matplotlib.pyplot as plt
		plt.figure(figsize=(8, 7))
		plt.imshow(cropped)
		plt.show(); plt.close()