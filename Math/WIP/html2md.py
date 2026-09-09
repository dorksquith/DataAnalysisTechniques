import sys
from html_to_markdown import ConversionOptions, convert

infname  = sys.argv[1]
outfname = infname.split('.html')[0]+".md"
with open(sys.argv[1]) as f: 
	html = f.read()

#html = "<h1>Hello</h1><p>This is <strong>formatted</strong> content.</p>"
	options = ConversionOptions(
	    heading_style="atx",
	    list_indent_width=2,
	)
	result = convert(html, options)
	markdown = result.content

	try:
		with open(outfname, "x", encoding="utf-8") as f:
			f.write(markdown)
	except FileExistsError:
		print(f"{outfname} already exists, exclusive creation aborted.")