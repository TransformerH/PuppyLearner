def bb_intersection_over_union(boxA, boxB):
    iou = 0;
    Alx = boxA[0]; Aly = boxA[1]; Arx = boxA[0] + boxA[2]; Ary = boxA[1] + boxA[3];
    Blx = boxB[0]; Bly = boxB[1]; Brx = boxB[0] + boxB[2]; Bry = boxB[1] + boxB[3];
    if Alx > Brx or Blx > Arx:
        return iou;
    if Ary < Bly or Bry < Aly:
        return iou;
    areaA = boxA[2] * boxA[3];
    areaB = boxB[2] * boxB[3];
    areaI = (min(Arx, Brx) - max(Alx, Blx)) * (min(Ary, Bry) - max(Aly, Bly));
    return areaI / (areaA + areaB - areaI)


# IOU = 1 ---  LOSS = 0
# IOU = 0 --- LOSS = 1
# GIoU Algorithm: https://giou.stanford.edu/
def GIOU(boxP, boxG):
    Plx = boxP[0]; Ply = boxP[1]; Prx = boxP[0] + boxP[2]; Pry = boxP[1] + boxP[3];
    PxMax = max(Plx.item(), Prx.item()); PxMin = min(Plx.item(), Prx.item());
    PyMax = max(Ply.item(), Pry.item()); PyMin = min(Ply.item(), Pry.item());

    Glx = boxG[0]; Gly = boxG[1]; Grx = boxG[0] + boxG[2]; Gry = boxG[1] + boxG[3];
    GxMax = max(Glx.item(), Grx.item()); GxMin = min(Glx.item(), Grx.item());
    GyMax = max(Gly.item(), Gry.item()); GyMin = min(Gly.item(), Gry.item());

    # calculate area
    gt_area = (GxMax - GxMin) * (GyMax - GyMin)
    pr_area = (PxMax - PxMin) * (PyMax - PyMin)

    #calculate intersaction
    IxMax = max(PxMin, GxMin); IxMin = min(PxMax, GxMax);
    IyMax = max(PyMin, GyMin); IyMin = min(PyMax, GxMax);

    in_area = 0
    if IxMin > IxMax and IyMin > IyMax:
        in_area = (IxMin - IxMax) * (IyMin - IyMax);

    # finding the coordinate of smallest enclosing box
    CxMin = min(PxMin, GxMin); CxMax = max(PxMax, GxMax);
    CyMin = min(PyMin, GyMin); CyMax = max(PyMax, GyMax);

    # Calculate area of smallest enclosing box
    enclosedBox_area = (CxMax - CxMin) * (CyMax - CyMin);

    union_area = gt_area + pr_area - in_area
    IoU = in_area / union_area

    GIoU = IoU - (enclosedBox_area - union_area) / enclosedBox_area
    Loss_IoU = 1 - IoU
    Loss_GIoU = 1- GIoU
    return Loss_GIoU, Loss_IoU

def bbox_loss(boxA, boxB):
    giou_list = []
    iou_list = []
    for i in range(len(boxA)):
        GIoU_loss, IoU_loss = GIOU(boxA[i], boxB[i])
        giou_list.append(GIoU_loss)
        iou_list.append(IoU_loss)
    return giou_list, iou_list

