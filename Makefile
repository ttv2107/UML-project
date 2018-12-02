.PHONY: zipdata
zipdata:
	zip -r data.zip data/

.PHONY: getdata
getdata:
	rm -rf data/
	unzip data.zip
