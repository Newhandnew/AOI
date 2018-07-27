import xml.etree.cElementTree as ET


def get_defect_list_from_xml(xml_file):
    root = ET.ElementTree(file=xml_file).getroot()
    panel_info = root.find('PanelDefectInfo')
    pattern_info = panel_info.find('PatternDefectInfos')
    defect_list = []
    for pattern in pattern_info:
        # image_name = pattern.find('ImageFilename')
        # print(image_name.text)
        defect_info = pattern.find('DefectInfos')
        for defect in defect_info:
            bounding_box = defect.find('BoundBox')
            x = int(bounding_box.find('x').text)
            y = int(bounding_box.find('y').text)
            defect_point = (x, y)
            defect_list.append(defect_point)

    return defect_list


def main():
    xml_file = '/media/new/A43C2A8E3C2A5C14/Downloads/AOI_dataset/0703/NG/4A835M81SSZZ_remarked.xml'
    # tree = ET.ElementTree(file=xml_file)
    # root = tree.getroot()
    # for child in root:
    #     print('child tag: {}, child attrib: {}, child text: {}'.format(child.tag, child.attrib, child.text))
    #     for sub in child:
    #         print('sub tag: {}, sub attrib: {}, sub text: {}'.format(sub.tag, sub.attrib, sub.text))
    #
    # for test in root.find('PanelDefectInfo'):
    #     print('test tag: {}, test attrib: {}, test text: {}'.format(test.tag, test.attrib, test.text))

    defect_list = get_defect_list_from_xml(xml_file)

    print(defect_list)


if __name__ == '__main__':
    main()