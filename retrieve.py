"""
This function downloads images from Google.
"""

import os
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
from apiclient.discovery import build

def search_images(query, start=1, num_images=10, colorType=None, dominantColor=None, api_key=None, id=None):
    resource = build("customsearch", 'v1', developerKey=api_key).cse()
    max_num_results = 10
    
    # There is an implicit range for custom search, values must be between [1, 201]
    if num_images + start > 201:
        num_images = 201 - start

    items = []
    if num_images <= max_num_results:
        results = resource.list(
            q=query, 
            cx=id, 
            searchType="image", 
            start=start,
            num=num_images,
            imgColorType=colorType,
            imgDominantColor=dominantColor
        ).execute()
        items = results['items']
    else:
        
        for i in range(start, num_images, max_num_results):
            results = resource.list(
                q=query, 
                cx=id, 
                searchType="image", 
                start=i,
                num=max_num_results,
                imgColorType=colorType,
                imgDominantColor=dominantColor
            ).execute()
            items += results['items']
    response = []
    for it in items:
        response.append({
            "url": it["link"],
            "caption": it["title"]
        })
    return response

def retrieve(target_name, outpath, num_class_images):
    num_images = 2*num_class_images

    if len(target_name.split()):
        target = '_'.join(target_name.split())
    else:
        target = target_name
    os.makedirs(f'{outpath}/images', exist_ok=True)

    if len(list(Path(f'{outpath}/images').iterdir())) >= num_class_images:
        return

    results = search_images(target_name, num_images=num_images)
    count, urls, captions = 0, [], []

    for each in results:
        name = f'{outpath}/images/{count}.jpg'
        success = True
        while True:
            try:
                img = requests.get(each['url'], timeout=3)
                success = True
                break
            except:
                print("Connection refused by the server..")
                success = False
                break
        if success and img.status_code == 200:
            print(len(img.content), count)
            try:
                _ = Image.open(BytesIO(img.content))
                with open(name, 'wb') as f:
                    f.write(img.content)
                urls.append(each['url'])
                captions.append(each['caption'])
                count += 1
            except Exception as e:
                print(e)
                print("not an image")
        if count > num_class_images:
            break

    with open(f'{outpath}/captions.txt', 'w') as f:
        for each in captions:
            f.write(each.strip() + '\n')

    # with open(f'{outpath}/images.txt', 'w') as f:
    #     for p in range(count):
    #         f.write(f'{outpath}/{target}/{p}.jpg' + '\n')
