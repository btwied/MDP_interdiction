#!/usr/bin/python

from sys import argv
from json import dumps

def get_data_string(filename):
	"Reads the file then removes unnecessary comments and whitespace."
	with open(filename) as f:
		data = f.read()
	# remove comments and compress whitespace
	data = " ".join(" ".join(filter(None, map(lambda l: l if "//" not in l \
			else l[:l.index("//")], data.split("\n")))).split())
	return data

def match_parens(data_str, paren_type="{}"):
	"""
	Recursively search a string for matched parens.
	Returns a dict mapping labels (before parens) to contents (inside parens).

	Each key in the return dict is a portion of data_str preceeding a pair of
	top-level parens. Each value is the result of calling match_parens() on the
	portion of data_str inside the parens. If no matched parens are found, then
	{data_str:None} is returned.

	The first open paren and its match define the following split:
	data_str = label +"{"+ contents +"}"+ rest

	The return dict has approximately this form:
	{label : match_parens(contents), match_parens(rest)}

	Example input:
	data_str = "aa{bb{cc}}{dd}ee{}[ff]", paren_type="{}"
	Exaple output:
	{"aa":{"bb":"cc"},"":"dd","ee":"","[ff]":None}
	"""
	open_paren = paren_type[0]
	close_paren = paren_type[1]

	i = data_str.find(open_paren)
	if i < 0: # no matching parens found
		return data_str

	opens = 0
	for j in range(i, len(data_str)): # find matching close paren index
		if data_str[j] == open_paren:
			opens += 1
		elif data_str[j] == close_paren:
			opens -= 1
		if opens == 0:
			break
	if data_str[j] != close_paren: # no matching parens found
		return data_str

	label = data_str[:i].strip("; ")
	contents = data_str[i+1:j].strip("; ")
	rest = data_str[j+1:].strip("; ")

	match_dict = {label : match_parens(contents, paren_type)}
	rest_match = match_parens(rest, paren_type)
	if isinstance(rest_match, str):
		if len(rest) > 0:
			match_dict[rest] = None
	else:
		for key,value in rest_match.items():
			count = 2
			rest_label = key
			while rest_label in match_dict: # avoid overwriting
				rest_label = key + "_occurence_#" + str(count)
				count += 1
			match_dict[rest_label] = value
	return match_dict

if __name__ == "__main__":
	data_str = get_data_string(argv[1])
	braces_dict = match_parens(data_str, "{}")
	print dumps(braces_dict, indent=2)
