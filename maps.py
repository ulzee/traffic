
from time import sleep
import gmplot
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from glob import glob
import cv2
import json
import numpy as np
from utils import *
from parula import *

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
	sleep(1.5)
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
	ints=None, # custom color intensities based on cmap
	cmap=parula_map,
	lines=None, colors=None, zoom=12, show=False, crop=60,
	wait=1.5,
	opacity=0.5,
	edge=2,
	scatter=False,
	key=None,
	coords_file='/home/ubuntu/traffic/data/stop_coords.json'):

	_, radj = reverse_graph(vs, adj)

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
			lines.append((seg, np.array(line)))
	gcoords = np.array(list(stops.values()))

	overlay = np.zeros((256, 256, 4)).astype(np.uint8)
	overlay[:,...,:3] = 255
	overlay[:,...,-1] = 255
	cv2.imwrite('overlay.png', overlay)

	clan, clng = np.mean(gcoords, axis=0)

	with open('../.keys.txt') as fl:
		gmkey = fl.read()

	gmap = gmplot.GoogleMapPlotter(
		clan,
		clng,
		zoom, gmkey)

	bounds_dict = dict(north=40.9,south=40.7,west=-75.0,east=-73)
	gmap.ground_overlay('https://i.imgur.com/kEIlxLf.png', bounds_dict, opacity=opacity)
	assert len(gmap.ground_overlays)
	slats, slngs = zip(*gcoords)
	if scatter:
		gmap.scatter(
			slats, slngs,
			'cornflowerblue',
			size=20, marker=False, face_alpha=1)



	for segname, line in lines:
		lats, lngs = zip(*line)
		clr = '#555555' if ints is not None else 'cornflowerblue'
		# if ints is not None:
		# 	if segname in ints:
		# 		rgb = cmap(ints[segname])
		# 		rgb = (255 * np.array(rgb[:-1])).astype(np.uint8).tolist()
		# 		clr = '#%02x%02x%02x' % tuple(rgb)
		gmap.plot(lats, lngs, clr, edge_width=edge)

	subd = 15 # 3 color subdivisions per segment
	if ints is not None:
		if segname in ints:
			for segname, line in lines:
				r1 = ints[segname]
				# r1 = 0
				before = [ints[child] for child in radj[segname]]
				r0 = np.mean(before) if len(before) else r1
				after = [ints[child] for child in adj[segname]]
				r2 = np.mean(after) if len(after) else r1
				# r0 = 1
				# r2 = 1

				for si in range(subd):
					interp = (si+1) / subd
					# print(segname, si, interp)
					assert interp <= 1.0 and interp >= 0
					cinterp = interp - (1/subd/2)
					if cinterp > 0.5:
						val = (r2 - r1) * ((cinterp - 0.5) / 0.5) + r1
					else:
						val = (r1 - r0) * (cinterp / 0.5) + r0
					rgb = cmap(val)
					rgb = (255 * np.array(rgb[:-1])).astype(np.uint8).tolist()
					clr = '#%02x%02x%02x' % tuple(rgb)
					lats, lngs = zip(*line)
					lat0, latf = (lats[1] - lats[0]) * (si/subd) + lats[0], (lats[1] - lats[0]) * interp + lats[0]
					lng0, lngf = (lngs[1] - lngs[0]) * (si/subd) + lngs[0], (lngs[1] - lngs[0]) * interp + lngs[0]
					# if si == 1:
					gmap.plot([lat0, latf], [lng0, lngf], clr, edge_width=3)

	gmap.draw( 'temp.html' )


	chrome_options = Options()
	chrome_options.add_argument("--headless")
	chrome_options.add_argument("--no-sandbox")

	driver = webdriver.Chrome(
		options = chrome_options)


	driver.get('file:///home/ubuntu/traffic/eval/temp.html')
	sleep(wait)
	driver.save_screenshot('%s.png' % name)
	driver.quit()

	img = cv2.cvtColor(cv2.imread('%s.png' % name), cv2.COLOR_BGR2RGB)
	cropped = img[crop:-crop, crop:-crop]
	cv2.imwrite('%s.png' % name, cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))

	if show:
		import matplotlib.pyplot as plt
		plt.figure(figsize=(8, 7))
		plt.imshow(cropped)
		plt.show(); plt.close()