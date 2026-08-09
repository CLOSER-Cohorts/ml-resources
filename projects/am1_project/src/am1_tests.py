from src.ml_resources import (
    read_dataset_from_file
)
from projects.am1_project.api import api_client
from src.ml_resources.data import colectica_utility

colectica_client = colectica_utility.C

raw_data_filename = './projects/am1_project/data/all_raw_data/all_raw_data_1.pickle'
all_raw_data=read_dataset_from_file(raw_data_filename)

items_with_no_topics={}
text_labels=[]
item_categories=[]
item_types=[]
has_categories=[]
count=0
for k, v in all_raw_data['all_data'].items():
   print(count)
   count=count+1
   if 'Topic' not in v.keys():
       items_with_no_topics[k]=v
       text_labels.append(v['TextLabel'])
       if v['ItemCategories']==[]:
          item_categories.append("")
          has_categories.append("no")
       else: 
          item_categories.append(v['ItemCategories'])
          has_categories.append("yes")
       item_types.append(colectica_client.item_code_inv(v['ItemType']))



request_body={
    "TextLabel": text_labels,
    "ItemCategories": item_categories,
    "ItemType": item_types,
    "HasCategories": has_categories
}

topic_classifications=api_client.execute_query(request_body)

