import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from xml.etree.ElementTree import Element


def TLSort(BBoxs):
    class BBox(object):
        '''
        BBox 类
        输入[x1,y1,w,h]
        接受 bbox 和 index作为BBox编号
        类中含有 text 变量 用于存放该bbox的文本行识别结果，请自助添加
        '''

        def __init__(self, bbox, index, text=''):
            self.bbox = bbox
            self.x1 = bbox[0]
            self.y1 = bbox[1]

            self.w = bbox[2]
            self.h = bbox[3]

            self.x2 = self.x1 + self.w
            self.y2 = self.y1 + self.h

            self.index = index
            self.row = list()
            self.col = list()
            self.rowspan = 0
            self.colspan = 0

            self.text = text or ''

            self.xmid = self.getxmid()
            self.ymid = self.getymid()
            self.printed = 0

        def getxmid(self):
            return self.x1 + self.w/2

        def getymid(self):
            return self.y1 + self.h/2


    def horizontallyconnected(source, target):
        if source.ymid > target.y1 and source.ymid < target.y2:
            return True
        else:
            return False

    BBoxPair = dict()
    BBoxlist =list()
    for i,bbox in enumerate(BBoxs):
        BBoxlist.append(BBox(bbox,i))
        BBoxPair[str(bbox)] = i
    all=BBoxlist
    linelist = list()
    indexlist = list()            
    for bbox in all:
        line = list()
        indexs = list()
        # removelist = list()
        line.append(bbox.bbox)
        indexs.append(bbox.index)
        # removelist.append(bbox)
        EXIST_FLAG = 0
        for target in all:
            if horizontallyconnected(bbox,target):
                if bbox is not target:
                    line.append(target.bbox)
                    indexs.append(target.index)
        indexs.sort()

        if len(indexlist) > 0:
            # print('len(indexlist) >0')
            for existed in indexlist:                           
                
                if existed == indexs:     
                    EXIST_FLAG = 1
                    break
            # print(EXIST_FLAG)
            if not EXIST_FLAG:
                linelist.append(line)    
                indexlist.append(indexs) 
        else:
            linelist.append(line)    
            indexlist.append(indexs)  

    # print(linelist)
    sortedlinelist = sorted(linelist, key= lambda x: np.array(x)[:,1].mean(), reverse=False)
    rowmatched = list()
    for line in sortedlinelist:
        # print(line)
        line = sorted(line, key= lambda x: x[0], reverse=False)
        for i in range(len(line)):
            line[i] = tuple(line[i])
        # print(line)
        rowmatched.append(line)
    # print(colmatched)
    rowmatched = list(filter(lambda f: not any(set(f) < set(g) for g in rowmatched), rowmatched))
    for row in rowmatched:
        for i in range(len(row)):
            row[i] = list(row[i])
    
    rowmatched = sum(rowmatched,[])
    output = []
    for i in rowmatched:
        output.append(BBoxPair[str(i)])
    # print(BBoxPair)
    return output

def __indent(elem, level=0):
    i = "\n" + level*"\t"
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            __indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def convert_2p_to_4p(Coords):
    # import pdb;pdb.set_trace()
    left_top   = str(int(float(Coords[0]))) + ',' + str(int(float(Coords[1])))
    right_top  = str(int(float(Coords[2]))) + ',' + str(int(float(Coords[1])))
    right_down = str(int(float(Coords[2]))) + ',' + str(int(float(Coords[3])))
    left_down  = str(int(float(Coords[0]))) + ',' + str(int(float(Coords[3])))
    cell_points = left_top + ' ' + right_top + ' ' + right_down + ' ' + left_down
    return cell_points

def save_per_json(eval_mesg, suffix):
    cells = []
    filename = eval_mesg[0]
    save_dir = os.path.join(eval_mesg[1], filename.replace(suffix, '.json'))

    for i in range(3, len(eval_mesg)):
        start_row = eval_mesg[i]['start_row']
        end_row = eval_mesg[i]['end_row']
        start_col = eval_mesg[i]['start_col']
        end_col = eval_mesg[i]['end_col']
        idx = eval_mesg[i]['id']
        cell_points = convert_2p_to_4p(eval_mesg[i]['Coords'])
        per_cell = {
            "id":int(idx),
            "tex":"",
            "content":[],
            "start_row": int(start_row),
			"end_row": int(end_row),
			"start_col": int(start_col),
			"end_col": int(end_col),
            "cell_points":cell_points  # 用于匹配内容
        }
        cells.append(per_cell)
    json_cells = {"cells": cells}

    with open(save_dir, "w") as f_target:
        json.dump(json_cells, f_target)
    # print('ok')

def save_per_xml(eval_mesg):
    filename = eval_mesg[0]
    # save_dir =os.path.join(eval_mesg[1], filename.replace('.jpg', '.xml'))
    save_dir =os.path.join(eval_mesg[1], filename.replace(os.path.splitext(filename)[-1], '.xml'))

    root = ET.Element('document', {'filename': filename})
    tree = ET.ElementTree(root)

    tableid = ET.Element('table', {'id': '1'})
    root.append(tableid)

    # table coords
    h,w = eval_mesg[2]
    tb_points = '0,0 ' + str(w)+','+'0 ' + str(w)+','+str(h)+' ' + '0,'+str(h)
    coords = Element("Coords", {'points': tb_points})
    tableid.append(coords)

    for i in range(3, len(eval_mesg)):
        start_row = eval_mesg[i]['start_row']
        end_row = eval_mesg[i]['end_row']
        start_col = eval_mesg[i]['start_col']
        end_col = eval_mesg[i]['end_col']
        idx = eval_mesg[i]['id']
        cell = Element("cell", {'end-col': end_col, 'end-row': end_row, 'id': idx, 'start-col': start_col, 'start-row': start_row})

        cell_points = convert_2p_to_4p(eval_mesg[i]['Coords'])
        coords = Element("Coords", {'points': cell_points})
        content = Element("content")
        content.text = "chenbangdong"
        
        cell.append(coords)
        cell.append(content)

        tableid.append(cell)

    __indent(root)
    tree.write(save_dir, encoding='utf-8', xml_declaration=True)
    # print("Finish ", name)

def save_per_html(eval_mesg):
    cells = []
    # wrong_list = []
    maxrows, maxcols = 0, 0
    for i in range(0, len(eval_mesg)):
        sr = int(eval_mesg[i]['start_row'])
        er = int(eval_mesg[i]['end_row'])
        sc = int(eval_mesg[i]['start_col'])
        ec = int(eval_mesg[i]['end_col'])
        # idx = eval_mesg[i]['id']
        # cell_points = convert_2p_to_4p(eval_mesg[i]['Coords'])

        maxrows,maxcols = max(maxrows,er),max(maxcols,ec)
        cells.append((sr,sc,er,ec))
    maxrows+=1
    maxcols+=1
    caps = np.zeros((maxrows,maxcols)) 
    cell_map = np.zeros((maxrows,maxcols)) 
    num = 0
    for item in cells:
        num+=1
        sr,sc ,er,ec = item[0:4]
        # print(sr,sc ,er,ec)
        for row in range(sr,er+1):
            if(ec<sc): ec=sc
            cell_map[row][sc:ec+1]= num
    # print(cell_map)
    tmp = np.where(cell_map==0)[0]

    if False:
    # if tmp.shape[0]!=0:
        # wrong_list.append(id)
        # import pdb;pdb.set_trace()
        print(eval_mesg)
        return '<html><body><table><tr><td></td></tr></table></body></html>'
    else:
        struct = '<html><body><table>'
        for row in range(maxrows):
            col = 0
            col = checkcol(caps,row,col)
            struct += '<tr>'
            while(col<maxcols):  
                row_span, col_span = span_num(cell_map,row,col)
                if row_span == 1 and col_span == 1:
                    struct += '<td>'
                else:
                    struct += '<td'
                    if(row_span!=1):                            #这两个可能需要根据情况调整顺序c
                        s = " rowspan=\""+str(row_span)+"\""        #根据需要看加不加空格，加空格是因为TEDS的评估要把struct处理成序列
                        struct += s
                    if(col_span!=1):
                        s = " colspan=\""+str(col_span)+"\""
                        struct += s
                    struct += '>'

                caps = occupy(caps, row, col, row_span, col_span, 1)
                col = checkcol(caps,row,col)
                struct += '</td>'
            struct += '</tr>'
        struct += '</table></body></html>'
    return struct

class NorString:
    def __init__(self, Q2B=True, map_dict=None, delete_blank=False, retain_1_blank=False, logger=None):
        self.Q2B = Q2B
        self.map_dict = map_dict
        self.delete_blank = delete_blank
        self.retain_1_blank = retain_1_blank
        # self.logger = logger
        # self.logger.debug('not string mapping: {}'.format(map_dict))

    def __call__(self, string):
        source_string = string
        if self.Q2B:
            string = strQ2B(string)
        if self.map_dict is not None:
            new_string = ''
            for c in string:
                if c in self.map_dict.keys():
                    new_string += self.map_dict[c]
                else:
                    new_string += c
            string = new_string
        if self.delete_blank:
            string = string.replace(' ', '')
        elif self.retain_1_blank:
            while '  ' in string:
                string = string.replace('  ', ' ')
        # if self.logger is not None and string != source_string:
        #     self.logger.debug('[source]{}\n[new   ]{}'.format(source_string, string))
        return string

char_map_table = {
    '—': '-',
    '﹣': '-',
    '―': '-',
    '–': '-',
    '▁': '_',
    '﹢': '+',
    '¥': '￥',
    '●': '·',
    '•': '·',
    '▪': '·',
    '⋅': '·',
    '∙': '·',
    '✱': '*',
    '✻': '*',
    '✽': '*',
    '↩': '↲',
    '↵': '↲',
    '‘': '\'',
    '’': '\'',
    '′': '\'',
    '“': '"',
    '”': '"',
    '〝': '"',
    '〞': '"',
    '◯': '○',
    '☐': '□',
    '∶': ':',
    '︰': ':',
    '✔': '√',
    '✓': '√',
    '內': '内',
    'ˆ': '^',
    'С': 'C',
    'А': 'A',
    '‥': '..',
    '％': '%'
}

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def save_per_html_with_ocr(eval_mesg):
    cells = []
    # wrong_list = []
    maxrows, maxcols = 0, 0
    train_transform_label = NorString(Q2B=True, map_dict=char_map_table, 
                                    delete_blank=True, retain_1_blank=True, logger=None)
    logic_dist = {}
    for i in range(3, len(eval_mesg)):
        sr = int(eval_mesg[i]['start_row'])
        er = int(eval_mesg[i]['end_row'])
        sc = int(eval_mesg[i]['start_col'])
        ec = int(eval_mesg[i]['end_col'])
        # idx = eval_mesg[i]['id']
        # cell_points = convert_2p_to_4p(eval_mesg[i]['Coords'])

        '''------ 增加ocr结果来生成HTML带有内容的GT -----'''
        TLs = eval_mesg[i]['TL']
        if len(TLs) == 1:
            cell_tokens = TLs[0]['TL_tokens']
            cell_tokens = cell_tokens.replace('<g></g>', '')  # 不考虑图片
            cell_tokens = train_transform_label(cell_tokens)  # 对齐识别与GT
        elif len(TLs) == 0:
            cell_tokens = ''
        else:
            BBoxs = []
            for TL_item in TLs:
                TL_bbox = np.array(TL_item['TL_bbox'])
                minx, maxx, miny, maxy = min(TL_bbox[:, 0]), max(TL_bbox[:, 0]), min(TL_bbox[:, 1]), max(TL_bbox[:, 1])
                BBoxs.append([minx, miny, maxx-minx, maxy-miny])
            rank = TLSort(BBoxs)
            cell_tokens = ''
            for n in rank:
                cell_tokens = cell_tokens + TLs[n]['TL_tokens']
                # cell_tokens = cell_tokens + TLs[n]['TL_tokens'] + ' '  # 换行符号
            cell_tokens = cell_tokens.replace('<g></g>', '')  # 不考虑图片
            cell_tokens = train_transform_label(cell_tokens)  # 对齐识别与GT
            # print(cell_tokens)
        logic_dist[str(sr)+','+str(sc)+','+str(er)+','+str(ec)] = cell_tokens
        '''-------------------------------------------'''

        maxrows,maxcols = max(maxrows,er),max(maxcols,ec)
        cells.append((sr,sc,er,ec,cell_tokens))
    maxrows+=1
    maxcols+=1
    caps = np.zeros((maxrows,maxcols)) 
    cell_map = np.zeros((maxrows,maxcols)) 
    num = 0
    for item in cells:
        num+=1
        sr,sc,er,ec = item[0:4]
        for row in range(sr,er+1):
            if(ec<sc): ec=sc
            cell_map[row][sc:ec+1]= num
    tmp = np.where(cell_map==0)[0]

    if tmp.shape[0]!=0:
        # wrong_list.append(id)
        print(eval_mesg)
        return '<html><body><table></table></body></html>'
    else:
        struct = '<html><body><table>'
        for row in range(maxrows):
            col = 0
            col = checkcol(caps,row,col)
            struct += '<tr>'
            while(col<maxcols):  
                row_span, col_span = span_num(cell_map,row,col)
                key = str(row)+','+str(col)+','+str(row+row_span-1)+','+str(col+col_span-1)
                try:
                    ocr = logic_dist[key]
                except:
                    print(eval_mesg)
                    return '<html><body><table></table></body></html>'
                if row_span == 1 and col_span == 1:
                    struct += '<td>'
                    struct += ocr
                else:
                    struct += '<td'
                    if(row_span!=1):                            #这两个可能需要根据情况调整顺序c
                        s = " rowspan=\""+str(row_span)+"\""        #根据需要看加不加空格，加空格是因为TEDS的评估要把struct处理成序列
                        struct += s
                    if(col_span!=1):
                        s = " colspan=\""+str(col_span)+"\""
                        struct += s
                    struct += '>'
                    struct += ocr
                caps = occupy(caps, row, col, row_span, col_span, 1)
                col = checkcol(caps,row,col)
                struct += '</td>'
            struct += '</tr>'
        struct += '</table></body></html>'
    return struct

def save_Honor_json_format(eval_mesg):
    cell_list = []
    for i in range(3, len(eval_mesg)):
        cell = {}
        cell["label"] = "cell"

        bbox = list(map(float, eval_mesg[i]['Coords']))
        cell["points"] = bbox
        
        sr = int(eval_mesg[i]['start_row'])
        er = int(eval_mesg[i]['end_row'])
        sc = int(eval_mesg[i]['start_col'])
        ec = int(eval_mesg[i]['end_col'])
        cell["rowStart"],cell["colStart"],cell["rowSpan"],cell["colSpan"] = sr+1, sc+1, er-sr+1, ec-sc+1
        cell["shape_type"] = "polygon"
        cell["group_id"]  = None
        cell["flags"]= {}
        cell_list.append(cell)
    result = {}
    result["version"]= "4.5.9"
    result["flags"]= {}
    result["shapes"] = cell_list
    result["imagePath"] = eval_mesg[0]
    return result

def format_unicode(json_path):
    with open(json_path, 'r') as f_json:
        json_res = json.load(f_json)
    with open(json_path, 'w',encoding='utf-8') as f_res:
        json.dump(json_res, f_res, ensure_ascii=False)
    return json_res

def span_num(caps,cur_row,cur_col):
    maxrow,maxcol = caps.shape[0],caps.shape[1]
    row_span,col_span = 0,0
    for row in range(cur_row,maxrow):
        if (caps[row][cur_col]!=caps[cur_row][cur_col]):
            row_span = row - cur_row
            break
        elif row == maxrow-1 :
            row_span = row - cur_row+1
    for col in range (cur_col,maxcol):
        if (caps[cur_row][col]!=caps[cur_row][cur_col]):
            col_span = col - cur_col
            break
        elif col == maxcol-1 :
            col_span = col - cur_col+1
    return row_span,col_span

def checkcol(caps, currow, curcol):
    '''
    检查该列是否已被其他跨行跨列单元格占用
    input：
    caps 占用矩阵
    currow 当前行
    curcol 当前列
    output：
    curcol 当前行合法列
    '''
    maxcols = len(caps[0])
    # print(currow,curcol)
    while curcol<maxcols and caps[currow][curcol]:
        curcol += 1
    return curcol

def occupy(caps, startrow, startcol, rowspan, colspan, id):
    '''
    填充占用矩阵
    input:
    caps
    startrow
    startcol
    rowspan
    colspan
    output:
    cap
    '''
    for i in range(startrow, startrow + rowspan):
        for j in range(startcol, startcol + colspan):
            caps[i][j] = 1
    return caps

if __name__ == '__main__':
    json_root = '/home/eecbd/dp4t/Table_Data/Honor/第二批数据/宣传彩页修改后-simplified'
    for name in tqdm(os.listdir(json_root)):
        json_path = os.path.join(json_root, name)
        format_unicode(json_path)