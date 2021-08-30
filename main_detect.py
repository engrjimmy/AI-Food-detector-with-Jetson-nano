import cv2
import numpy as np
import os
import pyrealsense2 as rs
import time
import label_image_class

font = cv2.FONT_HERSHEY_SIMPLEX
font_color = (0, 0, 255)

def get_masked_sections(im , crop_dim=0, bg_hue=93, erode_size = 0):

    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    lower = np.array([bg_hue, 33, 50])
    upper = np.array([180, 255, 255])

    # preparing the mask to overlay
    mask = cv2.inRange(hsv, lower, upper)

    mask = 255 - mask

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.ones(mask.shape) * 255

    sections = []

    for c in contours:
        if cv2.contourArea(c) > 400:
            # c = cv2.convexHull(c)
            mask = cv2.drawContours(mask ,[c] ,-1 ,0 ,cv2.FILLED ,1)
            # mask = cv2.dilate(mask, None, iterations=4)
            if erode_size > 0:
                mask = cv2.dilate(mask, None, iterations=erode_size)
            sections.append(cv2.boundingRect(c))


    mask = 255 - mask if mask[0][0] == 255 else mask

    mask = mask.astype('uint8')

    # mask = cv2.merge([mask,mask,mask])
    # mask = mask / 255
    # im = im * mask
    masked_im = cv2.bitwise_and(im,im,mask=mask)

    for s in sections:
        x, y, w, h = s
        input = im[y:y+h,x:x+w]
        cv2.imwrite('tmp.jpg',input)
        top_k,labels,results = label_image_class.classify('tmp.jpg')
        cv2.rectangle(masked_im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        masked_im = cv2.putText(masked_im, str(labels[top_k[0]])+':'+str(results[top_k[0]]), (x,y), font, 0.35, font_color, 1, cv2.LINE_AA)

    return masked_im

if __name__ == "__main__":

    label_image_class.init()

    realsense_ctx = rs.context()
    connected_devices = []
    '''
    for i in range(len(realsense_ctx.devices)):
       detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
       connected_devices.append(detected_camera)
    '''
    # ストリーム(Depth/Color)の設定
    config = rs.config()
    #config.enable_device(connected_devices[0])
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # ストリーミング開始
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # Alignオブジェクト生成
    align_to = rs.stream.color
    align = rs.align(align_to)

    time.sleep(2)

    try:
        while True:

            # フレーム待ち(Color & Depth)
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            # depth_frame = aligned_frames.get_depth_frame()
            # if not depth_frame or not color_frame:
            #     continue

            color_image = np.asanyarray(color_frame.get_data())
            # depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            color_image = color_image[135:398,172:611]
            # cv2.imshow('rs_stream', color_image)
            # break
            cv2.imshow('rs_stream',get_masked_sections(color_image))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        raise (e)
        # ストリーミング停止
        pipeline.stop()
