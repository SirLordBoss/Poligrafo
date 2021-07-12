import requests
import xml.etree.ElementTree as ET

#xml_test_contradiction_doc = requests.get("https://nlp.stanford.edu/projects/contradiction/RTE2_test_negated_contradiction.xml")
xml_dev_contradiction_doc = requests.get("https://nlp.stanford.edu/projects/contradiction/RTE2_dev_negated_contradiction.xml")
#xml_rte1_dev1_doc = requests.get("https://nlp.stanford.edu/projects/contradiction/RTE1_dev1_3ways.xml")


xml_real_life_contradiction_doc = requests.get("https://nlp.stanford.edu/projects/contradiction/real_contradiction.xml")

#with open('contradiction_dev_data.xml', 'ab') as f:
    #f.write(xml_dev_contradiction_doc.content)
    #f.write(xml_text_contradiction_doc.content)
    #print("done")


#with open('inbetween_real_life_data.xml', "wb") as f:
#    f.write(xml_real_life_contradiction_doc.content)

def filter_contradictions():
    tree = ET.parse('inbetween_real_life_data.xml')
    root = tree.getroot()

    for pair in root.findall('pair'):

        print("this is the type: " + pair.attrib['type'])
        if(pair.attrib['type'] != 'negation'):
            print("ain't a negation")
            root.remove(pair)
        #if(pair.attrib('type') != 'negation'):

    tree.write('output.xml') #just add this manually to the dataset

filter_contradictions()

