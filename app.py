from flask import Flask,request
from flask_cors import CORS, cross_origin
from pythainlp.corpus.common import thai_words
from pythainlp import Tokenizer,word_tokenize
import numpy as np
import requests
import json

app = Flask(__name__)

cors = CORS(app,resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'
# annotations = ocrmac.OCR('Cropped2.jpg').recognize()
# print(annotations)
# ocrmac.OCR('Cropped3.jpg').annotate_PIL()
pathbackend = requests.get('http://localhost:8081/api/database_path')
pathnodejs = 'http://localhost:8081'
# pathnodejs = 'https://api-fda.ponnipa.in.th'
# pathbackend = requests.get('https://api-fda.ponnipa.in.th/api/database_path')

# url = data.backend_path
backend_path = json.loads(pathbackend.text)
backend_path = backend_path["backend_path"]

@app.route('/')
def hello():
    return pathnodejs

@app.route('/checkkeyword', methods=["GET"])
def checkkeyword():
    name = request.args.get('name')
    # name = 'รายละเอียดสินค้าแท้% Fercy Fiber S เฟอร์ซี่ ไฟเบอร์ เอส Fercy Diet เฟอซี่ไดเอทFercy Diet เฟอร์ซี่ เคล็ดลับหุ่นดี คุมหิว อิ่มนาน น้ำหนักลงง่ายๆ ไม่ต้องอด ช่วยลดความอยากอาหาร ดักจับไขมัน'
    
    keyword_dicts = requests.get(pathnodejs+'/api/keyword_dicts?status=1')
    keyword_dicts = keyword_dicts.text
    keyword_dicts = json.loads(keyword_dicts)
    keyword_dicts = np.asarray(keyword_dicts)
    
    mapkeyword_dicts = requests.get(pathnodejs+'/api/keyword_dicts/mapdictId')
    mapkeyword_dicts = mapkeyword_dicts.text
    mapkeyword_dicts = json.loads(mapkeyword_dicts)
    mapkeyword_dicts = np.asarray(mapkeyword_dicts)
    
    array_mapkeyword_dicts = []
    for dict in mapkeyword_dicts:
        array_mapkeyword_dicts.append(dict['id'])
    
    
    array_keyword = []
    for restaurant in keyword_dicts:
        array_keyword.append(restaurant['name'])
    
    setting = requests.get(pathnodejs+'/api/token_setting')
    setting = setting.text
    setting = json.loads(setting)
    setting = np.asarray(setting)
    setting_front = int(setting[0]["front_space"])
    setting_back = int(setting[0]["back_space"])
    # name ='รายละเอียดสินค้ามัลติวิตพลัส อาหารเสริมเพิ่มน้ำหนัก สูตรใหม่ 2021 ล่าสุดMulti Vit Plus X10 สูตรพัฒนาใหม่ล่าสุดปรับปรุงจากสูตรเดิมให้ดีกว่า อย. 11-1-14859-5-0014แถมฟรี ตัวช่วยดูดซึมสารอาหาร 1'
    array_desc = tokenlist(name)
    arr_data = []
    spllist = []
    keywordarr = []
    
    # print(array_desc)
    for des in array_desc:
        # print(k['name'])
        for k in keyword_dicts:
            if k['name'] in des:
                keywordarr.append(k['id'])
    newkeywordarr = []
    # print('keywordarr',keywordarr)
    for ke in keywordarr:
        if ke not in newkeywordarr:
            newkeywordarr.append(ke) 
            
    idx = {x:i for i,x in enumerate(array_desc)}  
    # print(idx)
    idxdata = [] 
    for i,x in enumerate(array_desc):
        idxdata.append({'id':i,'x':x})
    # print(idxdata)
    tt = [idx[x] for x in array_keyword if x in idx]  
    tt=[]
    # print(array_desc)
    # print('array_keyword',array_keyword)
    for x in array_keyword:
        # print('x',x) 
        for i in idxdata:
            # print('x',i['x']) 
            # print('xx',x)
            if x in str(i['x']) and i['x'] != '':
                # print(i['x'])
                tt.append(i['id'])
    # print(tt)   
    tt.sort()
    new_list = []
    word = ''
    k = ''
    for w in tt:
        if w not in new_list:
            new_list.append(w) 
    
    # print('new_list',new_list)
    # print('array_keyword',array_keyword)  
    if len(new_list) == 0:
        i =0
        currentindex = 0
    else:
        i = new_list[0]
        currentindex = new_list[0]
    # print('idxdata',idxdata)
    # print('currentindex',new_list[0]) 
    
    # while i < descindex:
    for i in new_list:
        descindex = new_list[len(new_list)-1]
        if i >= currentindex and i in new_list and array_desc[i] != ' ' and currentindex<=descindex:

            backward = findbackward(array_desc,i,setting_front)
            forward = findforward(array_desc,i,setting_back)
            back = array_desc[backward:i]
            
            forw = array_desc[i:forward]
            currentindex = forward-1
            i=currentindex+1
            max_size = len(forw)
            last_index = max_size -1
            # print('len',forw[len(forw)-1])
            if forw[len(forw)-1] == ' ':
                forw.pop(last_index)
            # print('forw',forw)
            
            backw = ''.join(back)
            forwo = ''.join(forw)
            # print(f)
            sentent = backw+forwo
            
            
            sw = ''
            sen = sentent.split(' ')  
            dictarr = []
            status = 0
            # print('sen',sen)
            sen = [x for x in sen if x]
            
            for s in sen:
                # print(dictarr)
                if word != s:
                    sw += s +'  '
            # print('sw',sw)
            stradvertisetoken = tokenlist(sw)
            # print(stradvertisetoken)  
            
            dictstr = []
            myList = []
            myList = sw.split()
            arrdictid = []
            arrdictname = []
            # stradvertisetoken = [x.strip() for x in stradvertisetoken if x.strip()]
            # stradvertisetoken = [x for x in stradvertisetoken if x != '']
            print(stradvertisetoken)
            for s in stradvertisetoken:
                # print(s)
                if s != '  ' and s != "," and s != ' ' and s != ":" and s != '' and s != "[" and s != "]" and s != '️' and s != '"' and s !="'":
                    # print('s',s,len(s))
                    dictstr = requests.get(pathnodejs+'/api/dicts?name='+s)
                    dictstr = json.loads(dictstr.text)
                    dictstr = np.asarray(dictstr)
                    # print(dictstr)
                    if len(dictstr) > 0:
                        dictarr.append(int(dictstr[0]["id"]))
                        arrdictid.append(int(dictstr[0]["id"]))
                        arrdictname.append(s)
                    else:
                        indict =  "INSERT INTO dicts (id, name, status) VALUES (NULL,"
                        indict+= "'"+s +"',"
                        indict+= "'"+str(1) +"')"
                        # print(indict)
                        indictsql = requests.get(pathnodejs+'/api/dicts/createddicttoken?name='+ indict)
                        indictsql = json.loads(indictsql.text)
                        arrdictid.append(indictsql["id"])
                        arrdictname.append(s)
                        # print(indictsql["id"])
                    
            
            keyworddictId = []
            listfull = []
            for item in stradvertisetoken:
                if item in array_keyword:
                    listfull.append('<span style="color:red">'+item+'</span>')
                else:
                    listfull.append(item)
                    
            print('listfull',listfull)
            
            sumtext = listToString(listfull)
            # sumtext = sumtext.replace(' ','')

            print('sumtext',sumtext)
            sw = sumtext
            # print('status',status)
            dictkeyall= intersection(arrdictid, array_mapkeyword_dicts)

            arr_data.append({'dict_id':dictarr,
                             'dict_name':arrdictid,
            'keyword_dict_id':dictkeyall,
            'dict_name':arrdictname,
                             'sen':myList,
                             'sentent':sw})
            word = sen[len(sen)-1]
        else:
            i = i+1
            # currentindex = currentindex+1
                    
    return arr_data


# @app.route('/checkkeyword', methods=["GET"])
# def checkkeyword():
#     name = request.args.get('name')
#     id = request.args.get('id')
#     end = request.args.get('end')
#     # name = name.replace(' ', '')
#     # name = 'รายละเอียดสินค้าแท้% Fercy Fiber S เฟอร์ซี่ ไฟเบอร์ เอส Fercy Diet เฟอซี่ไดเอทFercy Diet เฟอร์ซี่ เคล็ดลับหุ่นดี คุมหิว อิ่มนาน น้ำหนักลงง่ายๆ ไม่ต้องอด ช่วยลดความอยากอาหาร ดักจับไขมัน'
#     deleteadvertise = "DELETE FROM advertise WHERE product_token_id = "+id
#     deleterule_based =  "DELETE FROM rule_based_keyword WHERE product_id = "+id
#     # print(deleterule_based)
#     # print(deleteadvertise)
#     requests.get(pathnodejs+'/api/rule_based/getbydict?name='+deleteadvertise)
#     requests.get(pathnodejs+'/api/rule_based/getbydict?name='+deleterule_based)
#     # product_data = requests.get('http://localhost:8081/api/products/getproductkeyword?start=1')
#     # product_data = product_data.text
#     # product_data = json.loads(product_data)
#     # print(product_data)
#     rulekey = requests.get(pathnodejs+'/api/rule_based_keyword/'+id)
#     # print(pathnodejs+'/api/rule_based_keyword/'+id)
#     rulekey = rulekey.text
#     rulekey = json.loads(rulekey)
#     # rulekey = np.asarray(rulekey)
#     # print(len(rulekey))
    
#     keyword_dicts = requests.get(pathnodejs+'/api/keyword_dicts?status=1')
#     keyword_dicts = keyword_dicts.text
#     keyword_dicts = json.loads(keyword_dicts)
#     keyword_dicts = np.asarray(keyword_dicts)
    
#     mapkeyword_dicts = requests.get(pathnodejs+'/api/keyword_dicts/mapdictId')
#     mapkeyword_dicts = mapkeyword_dicts.text
#     mapkeyword_dicts = json.loads(mapkeyword_dicts)
#     mapkeyword_dicts = np.asarray(mapkeyword_dicts)
    
#     array_mapkeyword_dicts = []
#     for dict in mapkeyword_dicts:
#         array_mapkeyword_dicts.append(dict['id'])
    
    
#     array_keyword = []
#     for restaurant in keyword_dicts:
#         array_keyword.append(restaurant['name'])
    
#     setting = requests.get(pathnodejs+'/api/token_setting')
#     setting = setting.text
#     setting = json.loads(setting)
#     setting = np.asarray(setting)
#     setting_front = int(setting[0]["front_space"])
#     setting_back = int(setting[0]["back_space"])
#     # name ='รายละเอียดสินค้ามัลติวิตพลัส อาหารเสริมเพิ่มน้ำหนัก สูตรใหม่ 2021 ล่าสุดMulti Vit Plus X10 สูตรพัฒนาใหม่ล่าสุดปรับปรุงจากสูตรเดิมให้ดีกว่า อย. 11-1-14859-5-0014แถมฟรี ตัวช่วยดูดซึมสารอาหาร 1'
#     array_desc = tokenlist(name)
#     arr_data = []
#     spllist = []
#     keywordarr = []
    
#     # print(array_desc)
#     for des in array_desc:
#         # print(k['name'])
#         for k in keyword_dicts:
#             if k['name'] in des:
#                 keywordarr.append(k['id'])
#     newkeywordarr = []
#     # print('keywordarr',keywordarr)
#     for ke in keywordarr:
#         if ke not in newkeywordarr:
#             newkeywordarr.append(ke) 
            
    
#     if len(keywordarr) > 0 and len(rulekey) == 0:
#         strkeywordarr = str(newkeywordarr).replace(' ','')
#         k =  "INSERT INTO rule_based_keyword (id, product_token_id, keyword_id) VALUES (NULL,"
#         k+= "'"+id +"',"
#         k+= "'"+strkeywordarr +"')"
#         # print(k)
#         # print(strkeywordarr)
        
#     idx = {x:i for i,x in enumerate(array_desc)}  
#     # print(idx)
#     idxdata = [] 
#     for i,x in enumerate(array_desc):
#         idxdata.append({'id':i,'x':x})
#     # print(idxdata)
#     tt = [idx[x] for x in array_keyword if x in idx]  
#     tt=[]
#     # print(array_desc)
#     # print('array_keyword',array_keyword)
#     for x in array_keyword:
#         # print('x',x) 
#         for i in idxdata:
#             # print('x',i['x']) 
#             # print('xx',x)
#             if x in str(i['x']) and i['x'] != '':
#                 # print(i['x'])
#                 tt.append(i['id'])
#     # print(tt)   
#     tt.sort()
#     new_list = []
#     word = ''
#     k = ''
#     for w in tt:
#         if w not in new_list:
#             new_list.append(w) 
    
#     # print('new_list',new_list)
#     # print('array_keyword',array_keyword)  
#     if len(new_list) == 0:
#         i =0
#         currentindex = 0
#     else:
#         i = new_list[0]
#         currentindex = new_list[0]
#     # print('idxdata',idxdata)
#     # print('currentindex',new_list[0]) 
    
#     # while i < descindex:
#     for i in new_list:
#         descindex = new_list[len(new_list)-1]
#         # print('descindex',descindex)
#         # currentindex = i+1
#         # print('i',i)
#         # print('currentindex',currentindex)
#         # print(array_desc[currentindex])
#         # print('new_list',new_list)
#         if i >= currentindex and i in new_list and array_desc[i] != ' ' and currentindex<=descindex:
            
#             # print('i',i)
#             # print('currentindex',currentindex)
#             # print(array_desc[currentindex])
#             backward = findbackward(array_desc,i,setting_front)
#             # if backward < currentindex and i!= new_list[0]:
#             #     b = currentindex
#             #     while b < descindex:
#             #         backward = findbackward(array_desc,b,setting_front)
#             #         if backward > currentindex:
#             #             b = backward
#             #             i = backward
#             #             currentindex = backward
#             #             print(backward)
#             #         b+=1
                        
                
#             # print('array_desc',array_desc[i])
#             # print('backward',backward)
#             # print('backward',array_desc[backward:i])
            
#             forward = findforward(array_desc,i,setting_back)
#             # print('forward',forward)
#             # print('forward',array_desc[i:forward])
            
#             back = array_desc[backward:i]
            
#             forw = array_desc[i:forward]
#             # print('back',back)
#             # print('forw',forw)
#             # print('test',forw[len(forw)-2])
#             # print('back',back[len(back)-1])
                
#             # print(arr_data[len(arr_data-1)])
#             # if len(arr_data) > 0:
#             #     # if back in arr_data[len(arr_data-1)]:
#             #         print('len',len(arr_data))
#             #         print(arr_data[int(len(arr_data))-1])
#             #         print(back)
#             #         if back :
                        
#             # if forw[len(forw)-2] == ' ':
#             #     forw = array_desc[i:forward-2]
#             #     currentindex = forward-2
#             #     i=currentindex+2
#             # else:
#             #     currentindex = forward+1
#             #     i=currentindex+1
#             currentindex = forward-1
#             i=currentindex+1
#             # print(currentindex)
#             # print(i)
#             # f = ''
#             # if forw[len(forw)-1] != ' ' or forw[len(forw)-2] != ' ':
#             #     f = array_desc[forward:len(array_desc)]
#             #     # print(f)
#             #     # print(f.index(' '))
#             #     f = f[0:f.index(' ')]
#             #     # print(f)
#             #     f = ''.join(f)
#             #     # forw = array_desc[i:forward+1]
            
#             max_size = len(forw)
#             last_index = max_size -1
#             # print('len',forw[len(forw)-1])
#             if forw[len(forw)-1] == ' ':
#                 forw.pop(last_index)
#             # print('forw',forw)
            
#             backw = ''.join(back)
#             forwo = ''.join(forw)
#             # print(f)
#             sentent = backw+forwo
            
            
#             sw = ''
#             sen = sentent.split(' ')  
#             dictarr = []
#             status = 0
#             # print(sen)
#             sen = [x for x in sen if x]
#             for s in sen:
#                 # print(dictarr)
#                 if word != s:
#                     sw += s +'  '
#             # print('sw',sw)
#             stradvertisetoken = tokenlist(sw)
#             # print(stradvertisetoken)  
#             dictstr = []
#             myList = []
#             myList = sw.split()
#             arrdictid = []
#             arrdictname = []
#             stradvertisetoken = [x.strip() for x in stradvertisetoken if x.strip()]
#             stradvertisetoken = [x for x in stradvertisetoken if x != '']
#             # print(stradvertisetoken)
#             for s in stradvertisetoken:
#                 # print(s)
#                 if s != '  ' and s != "," and s != ' ' and s != ":" and s != '' and s != "[" and s != "]" and s != '️' and s != '"' and s !="'":
#                     # print('s',s,len(s))
#                     dictstr = requests.get(pathnodejs+'/api/dicts?name='+s)
#                     dictstr = json.loads(dictstr.text)
#                     dictstr = np.asarray(dictstr)
#                     # print(dictstr)
#                     if len(dictstr) > 0:
#                         dictarr.append(int(dictstr[0]["id"]))
#                         arrdictid.append(int(dictstr[0]["id"]))
#                         arrdictname.append(s)
#                     else:
#                         indict =  "INSERT INTO dicts (id, name, status) VALUES (NULL,"
#                         indict+= "'"+s +"',"
#                         indict+= "'"+str(1) +"')"
#                         # print(indict)
#                         indictsql = requests.get(pathnodejs+'/api/dicts/createddicttoken?name='+ indict)
#                         indictsql = json.loads(indictsql.text)
#                         arrdictid.append(indictsql["id"])
#                         arrdictname.append(s)
#                         # print(indictsql["id"])
                    
            
#             # print('myList',myList)
#             # sql = str(dictarr).replace(' ','')
#             # rule =  "SELECT m.* FROM map_rule_based m WHERE m.status = 1 and m.dict_id = "
#             # rule+= "'"+sql +"'"
#             # # print(rule)
#             # rule_based = requests.get(pathnodejs+'/api/rule_based/getbydict?name='+rule)
#             # rule_based = json.loads(rule_based.text)
#             # rule_based = np.asarray(rule_based)
#             # # print(rule_based)
#             # if len(rule_based) > 0:
#             #     status = int(rule_based[0]["answer"])
#             keyworddictId = []
#             for mid in array_keyword:
#                 sw = sw.replace(mid,'<span style="color:red">'+mid+'</span>')
#             # print('dictarr',dictarr)
#             # print('myList',myList)
#             # print('sw',sw)
#             # print('status',status)
#             arr_data.append({'id':dictarr,
#                              'sen':myList,
#                              'sentent':sw})
#             stradvertise = str(dictarr).replace(' ','')
#             strarrdictid = str(arrdictid).replace(' ','')
#             strarrdictname = str(arrdictname).replace(' ','')
#             strarrdictname = str(strarrdictname).replace("'",'"')
#             # strmyList = str(myList).replace(' ','')
#             # print(myList)
            
#             Listsen = []
#             for m in myList:
#                 Listsen.append({"name":m})
#             # strmyList = str(strmyList).replace("'",'"')
#             # strmyList = str(strmyList).replace('["','[{"name":"')
#             # strmyList = str(strmyList).replace('"]','"}]')
#             # strmyList = str(strmyList).replace(",',","',',")
            
#             # strmyList = str(strmyList).replace(',','},{"name":')
#             # strmyList = str(strmyList).replace("']",'}]')
#             strmyList = str(Listsen)
#             strmyList = str(strmyList).replace(",'",',"')
#             strmyList = str(strmyList).replace("'",'"')
#             strmyList = str(strmyList).replace('""','"')
            
#             sw += "')"
#             sw= str(sw).replace("'  '","' ")
#             # print(sw)
             
#             print('arrdictid',arrdictid)
#             print('array_mapkeyword_dicts',array_mapkeyword_dicts)
#             dictkeyall = intersection(arrdictid, array_mapkeyword_dicts)
#             print('dictkeyall',dictkeyall)
#             finddictId = str(dictkeyall).replace(' ','') 
#             print('finddictId',finddictId)
#             print('strarrdictid',strarrdictid)
#             if len(keywordarr) > 0 and len(rulekey) == 0:
#                 ad =  "INSERT INTO advertise (id, product_token_id,keyword_dict_id, dict_id, dict_name, sen, sentent) VALUES (NULL,"
#                 ad+= "'"+id +"',"
#                 ad+= "'"+finddictId +"',"
#                 ad+= "'"+strarrdictid +"',"
#                 ad+= "'"+strarrdictname +"',"
#                 ad+= "'"+strmyList +"',"
#                 ad+= "'"+sw +""
#                 print(ad)
#                 addadvertise = requests.get(pathnodejs+'/api/rule_based/getbydict?name='+ad)
#                 # print(json.loads(addadvertise.text))
#             # print(arr_data)
#             word = sen[len(sen)-1]
#             # print(word)
#             # print('arr_data',arr_data)
#             # ws = sw.split(' ')  
#             # print(ws)  
#             # word = ws[len(ws)-1]
#             # print(word)
#             # if word == '':
#             #     word = ws[len(ws)-1]
#             # print('len(array_desc)',len(array_desc))
#             # print('backward',backward)
#             # print('forward',forward)
            
#             # print('word',word)
#         else:
#             i = i+1
#             # currentindex = currentindex+1
                    
#     return arr_data

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

# @app.route('/worktokendesc', methods=["GET"])
# @cross_origin(origin='*',headers=['Content-Type','Authorization'])
# def worktokendesc():
#     name = request.args.get('text')
#     # name = name.replace(' ', '')
#     # name = 'หรือหลังอาหารมื้อแรก ถ้ายังรู้สึกหิวปรับทาน 2 แคปซูล หลังอาหารมื้อแรกของวัน'
#     x = requests.get(pathnodejs+'/api/dicts?status=1')
#     dicts = x.text
#     dicts = json.loads(dicts)
#     words = set(thai_words())  # thai_words() returns frozenset
#     my_array = np.asarray(dicts)
    
#     for restaurant in my_array:
#         # print (restaurant['name'])
#         value = restaurant['name']
#         words.add(value) 
    
#     keyword_dicts = requests.get(pathnodejs+'/api/keyword_dicts?status=1')
#     keyword_dicts = keyword_dicts.text
#     keyword_dicts = json.loads(keyword_dicts)
#     keyword_dicts = np.asarray(keyword_dicts)
    
#     k = ''
#     key = []
#     for restaurant in keyword_dicts:
#         # print (restaurant['name'])
#         value = restaurant['name']
#         k+=restaurant['name']
#         key.append(restaurant['name'])
#         words.add(value) 
#     # print('k',k)
#     custom_tokenizer = Tokenizer(words)
#     name_result = custom_tokenizer.word_tokenize(name)
#     # namereal_result = custom_tokenizer.word_tokenize(k)
#     # print('name_result',name_result)
#     namereal_result = key
#     # print('namereal_result',namereal_result)
#     listfull=[]
#     for item in name_result:
#         na = ''
#         if item != ' ' and item != '(' and item != ')' and item != 'ผล':
#             # print(item)
#             # if any(word.startswith(item) for word in namereal_result):
#             for real in namereal_result:
#                 if item in real:
#                     listfull.append(item)
                    
                    
#     # print('listfull',listfull)
    
#     for k in key:
#         t = name.replace(k,'<span style="color:red">'+k+'</span>')
#         name = t

#     # print(t)
#     # print('name',name)
#     # array_desc = tokenlist(name)
    
#     return name


@app.route('/worktokendesc', methods=["GET"])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def worktokendesc():
    name = request.args.get('text')
    # name = name.replace(' ', '')
    # name = 'หรือหลังอาหารมื้อแรก ถ้ายังรู้สึกหิวปรับทาน 2 แคปซูล หลังอาหารมื้อแรกของวัน'
    x = requests.get(pathnodejs+'/api/dicts?status=1')
    dicts = x.text
    dicts = json.loads(dicts)
    words = set(thai_words())  # thai_words() returns frozenset
    my_array = np.asarray(dicts)
    
    for restaurant in my_array:
        # print (restaurant['name'])
        value = restaurant['name']
        words.add(value) 
    
    keyword_dicts = requests.get(pathnodejs+'/api/keyword_dicts?status=1')
    keyword_dicts = keyword_dicts.text
    keyword_dicts = json.loads(keyword_dicts)
    keyword_dicts = np.asarray(keyword_dicts)
    
    k = ''
    key = []
    for restaurant in keyword_dicts:
        # print (restaurant['name'])
        value = restaurant['name']
        k+=restaurant['name']
        key.append(restaurant['name'])
        words.add(value) 
    # print('k',k)
    custom_tokenizer = Tokenizer(words)
    name_result = custom_tokenizer.word_tokenize(name)
    # namereal_result = custom_tokenizer.word_tokenize(k)
    # print('name_result',name_result)
    namereal_result = key
    # print('namereal_result',namereal_result)
    listfull=[]
    dictkeyall = intersection(name_result, namereal_result)
    # print(dictkeyall)
    for item in name_result:
                if item in dictkeyall:
                    listfull.append('<span style="color:red">'+item+'</span>')
                else:
                    listfull.append(item)
                    
    # print('listfull',listfull)
    
    sumtext = listToString(listfull)
    # sumtext = sumtext.replace(' ','')
    # print(sumtext)

    # print(t)
    # print('name',name)
    # array_desc = tokenlist(name)
    
    return sumtext

def listToString(s):
     
    # initialize an empty string
    str1 = ""
 
    # return string
    return (str1.join(s))

def tokenlist(name):
    x = requests.get(pathnodejs+'/api/dicts?status=1')
    dicts = x.text
    dicts = json.loads(dicts)
    words = set(thai_words())  # thai_words() returns frozenset
    my_array = np.asarray(dicts)
    
    for restaurant in my_array:
        # print (restaurant['name'])
        value = restaurant['name']
        words.add(value) 
    
    keyword_dicts = requests.get(pathnodejs+'/api/keyword_dicts?status=1')
    keyword_dicts = keyword_dicts.text
    keyword_dicts = json.loads(keyword_dicts)
    keyword_dicts = np.asarray(keyword_dicts)
    
    array_keyword = []
    for restaurant in keyword_dicts:
        array_keyword.append(restaurant['name'])
        words.add(restaurant['name']) 
        
    custom_tokenizer = Tokenizer(words)
    name_result = custom_tokenizer.word_tokenize(name)
    
    array_desc = []
    for n in name_result:
        array_desc.append(n)
        
    return array_desc

def findbackward(array,index,setting):
    bc = 0
    mb = 1
    while bc < setting:
        cb = index-mb
        # print(cb)
        if cb < len(array):
            # print(array[cb])
            if array[cb] == ' ':
                bc = bc+ 1
            if cb == 0:
                bc = bc+ 1
        mb = mb+1
        # print('cb',cb)    
    return cb

def findforward(array,index,setting):
    # print('findforward',index)
    bc = 0
    mb = 1
    # print(bc)
    while bc < setting:
        # print(bc)
        cb = index+mb
        # print('cb',cb)
        if cb < len(array):
            # print(array[cb+3])
            if array[cb] == ' ':
                bc = bc + 1
                # 171
                # print(bc)
                # break;
        else:
            bc = bc + 1
        mb = mb+1
        # print('findforward',array[cb+1])
    return cb+1
 
if __name__ == "__main__":
    app.run(debug=False)