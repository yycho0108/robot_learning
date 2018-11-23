import cv2

def putText(frame, loc, txt, clr, h_align='c', v_align='c'):
    font = cv2.FONT_HERSHEY_DUPLEX
    font_size = 0.5
    ts = cv2.getTextSize(txt, font, font_size, 0)[0]

    dx = {'c':0.5,'l':0.0,'r':1.0}
    dy = {'c':0.0,'t':-1.0,'b':1.0}

    y = loc[1] - ts[1] * dy[v_align]
    x = loc[0] - ts[0] * dx[h_align]
    x,y = map(int, (x,y))
    pt = (x,y)
    cv2.rectangle(frame, (x,y-ts[1]), (x+ts[0],y), clr, -1)
    cv2.putText(frame, txt, (x,y), font, font_size, (255,255,255), 1)

def draw_bbox(frame, box, lbl, clr):
    H,W = frame.shape[:2]
    y1,x1,y2,x2 = [e*s for (e,s) in zip(box,[H,W,H,W])]
    x = int((x1+x2)/2)
    y = int((y1+y2)/2)
    w = int((x2-x1))
    h = int((y2-y1))

    cv2.rectangle(frame, (x-w//2, y-h//2), (x+w//2, y+h//2), clr, 2)
    putText(frame, (x1,y1), lbl, clr, h_align='l', v_align='t')
