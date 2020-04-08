import os, cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from gaussian.gaussian_model import SimpleGaussian
from logistic.logistic_model import LogisticRegression

class StopSignDetector():
    def __init__(self):
        '''
            Initilize your stop sign detector with the attributes you need,
            e.g., parameters of your classifier
        '''
        # choose which method to try
        # "SimpleGaussian", "NaiveBayes", or "LogisticRegression"
        self.method = "NaiveBayes"

        # initialize mean value for each class
        self.mean_dict = {}
        self.mean_dict['COLOR_RED']		= np.array([122.90029146,  36.63952687,  42.20470774])
        self.mean_dict['COLOR_BROWN']	= np.array([133.48214699, 100.11578226,  71.49936706])
        self.mean_dict['COLOR_OTHER']	= np.array([110.91446659, 120.69703685, 126.29357601])

        # initialize covariance matrix for each class
        self.cov_dict = {}
        self.cov_dict['COLOR_RED']  	= np.array([[3472.85255899, -189.07265668,   63.89804755],
                                                    [-189.07265668, 1106.75494927,  978.63599331],
                                                    [63.89804755,  978.63599331,  990.83383879]])
        self.cov_dict['COLOR_BROWN']	= np.array([[3875.35170522, 2855.30289674, 1720.8232254 ],
                                                    [2855.30289674, 2473.60440071, 1839.71033185],
                                                    [1720.8232254 , 1839.71033185, 1859.3362903 ]])
        self.cov_dict['COLOR_OTHER']	= np.array([[3287.09643207, 3056.21297624, 2879.15061134],
                                                    [3056.21297624, 3286.23666639, 3530.60415906],
                                                    [2879.15061134, 3530.60415906, 4653.45892218]])

        # initialize prior for each class
        self.prior_dict = {}
        self.prior_dict['COLOR_RED']	= 0.016864184071881394
        self.prior_dict['COLOR_BROWN']	= 0.04490358080450406
        self.prior_dict['COLOR_OTHER']	= 0.9371541010520053

        # initialize weights for logistic regression
        self.coeff = {}
        self.coeff['w'] = np.array([[0.01659144, -0.04449674, -0.03879027]])
        self.coeff['b'] = -0.00034812544815634343


    def compute_score(self, w, h, area_ratio, euler_num, img_size):
        '''
            Obtain a segmented image using a color classifier,
            e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
            call other functions in this class if needed

            Inputs:
                w           - width of the bounding box
                h           - height of the bounding box
                area_ratio  - the area of red pixel versus the area of the bounding box
                euler_num   - computed as number of objects (= 1) subtracted by number of holes (8-connectivity)
                img_size    - the size of the input image
            Outputs:
                score - an integer showing of how possible the region is stop sign
        '''
        score = 0
        # check the size of the bounding box
        if(w < img_size[1]/12 or h < img_size[0]/12):
            score -= 5
        if(w >= img_size[1]/10 and h >= img_size[0]/10):
            score += 1

        # check width versus height ratio 
        if(w/h < 1.4 and w/h > 0.7):
            score += 2
        else:
            score -= 2

        # check area ratio
        if(area_ratio >= 0.5 and area_ratio <= 0.8):
            score += 5
        else:
            score -= 5

        # check euler number 
        if(euler_num <= 0 and euler_num >= -9 or euler_num <= -15):
            score += 5
        elif(euler_num > -15 and euler_num < -9 or euler_num > 0):
            score -= 5
        
        return score


    def segment_image(self, img):
        '''
            Obtain a segmented image using a color classifier,
            e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        '''
        # create simple gaussian model for different classes
        model_red 		= SimpleGaussian(self.mean_dict['COLOR_RED'], self.cov_dict['COLOR_RED'])
        model_brown 	= SimpleGaussian(self.mean_dict['COLOR_BROWN'], self.cov_dict['COLOR_BROWN'])
        model_other 	= SimpleGaussian(self.mean_dict['COLOR_OTHER'], self.cov_dict['COLOR_OTHER'])

        # convert BGR to RGB
        # and reshape to (N,3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size = np.shape(img)
        img = np.reshape(img, (-1,3))

        # predict
        if(self.method == "SimpleGaussian"):
            pixel_red 	= model_red.predict(img)
            pixel_brown = model_brown.predict(img)
            pixel_other = model_other.predict(img)
        elif(self.method == "NaiveBayes"):
            pixel_red   = model_red.predict(img) * self.prior_dict['COLOR_RED']
            pixel_brown = model_brown.predict(img) * self.prior_dict['COLOR_BROWN']
            pixel_other = model_other.predict(img) * self.prior_dict['COLOR_OTHER']
        else:
            model = LogisticRegression(n_features=3)
            pred  = model.predict(img, self.coeff)

        # find max probability for each pixel
        tmp1    = pixel_red.tolist()
        tmp2    = pixel_brown.tolist()
        tmp3    = pixel_other.tolist()
        max_tmp = list(map(max, zip(tmp1, tmp2, tmp3)))
        max_tmp = np.array(max_tmp)

        # create mask image
        if(self.method == "SimpleGaussian" or self.method == "NaiveBayes"):
            # find max probability for each pixel
            tmp1    = pixel_red.tolist()
            tmp2    = pixel_brown.tolist()
            tmp3    = pixel_other.tolist()
            max_tmp = list(map(max, zip(tmp1, tmp2, tmp3)))
            max_tmp = np.array(max_tmp)
            mask_img = 255*np.array(max_tmp == pixel_red).astype('uint8')
        else:
            mask_img = 255 * pred.astype('uint8')

        # perform morphological operations
        mask_img = cv2.dilate(mask_img, (25,25), iterations = 10)
        mask_img = cv2.erode(mask_img, (25,25), iterations = 10)
        mask_img = mask_img.reshape(img_size[:2])

        return mask_img


    def get_bounding_box(self, img):
        '''
            Find the bounding box of the stop sign
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                is from left to right in the image.

            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        '''
        # get the masked image
        mask_img  = self.segment_image(img)
        img_size  = np.shape(mask_img)
        boxes     = []
        label_img = label(mask_img)
        regions   = regionprops(label_img)
        
        # check each regions in the input image
        for props in regions:
            box = props.bbox
            x, y, w, h = box[1], box[0], box[3]-box[1], box[2]-box[0]
            area_ratio = props.extent
            euler_num  = props.euler_number

            # compute the similarity score
            score = self.compute_score(w, h, area_ratio, euler_num, img_size)
            if(score >= 3):
                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)
                boxes.append([x,img_size[0]-y-h,x+w,img_size[0]-y])

        # get contours of each regions
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h  = cv2.boundingRect(contour)
            area        = cv2.contourArea(contour)
            
            # shortlisting the regions based on the area
            if(area > (img_size[1]/25) * (img_size[0]/25)):
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

                # check polygons
                if(len(approx)%4==0 and len(approx)!=4):
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)
                    box = [x,img_size[0]-y-h,x+w,img_size[0]-y]

                    # if current bounding box is not in the list
                    if(box not in boxes):
                        boxes.append(box)

        # the order of bounding boxes in the list is from left to right in the image
        if(len(boxes) > 1):
            boxes.sort(key = lambda x: x[0])

        return boxes



if __name__ == '__main__':
    folder = "trainset"
    my_detector = StopSignDetector()

    #################### Test Single Image ####################
    # img = cv2.imread('trainset/22.png')

    # #Display results:
    # #(1) Segmented images
    # mask_img = my_detector.segment_image(img)
    # mask_img = cv2.resize(mask_img, (600, 400))
    # cv2.imshow('masked image', mask_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # #(2) Stop sign bounding box
    # boxes = my_detector.get_bounding_box(img)
    # img = cv2.resize(img, (600, 400))
    # #fig = plt.figure()
    # #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # #plt.imshow(img)
    # cv2.imshow('BBox', img)
    # #plt.savefig("s.png", dpi=100, bbox_inches='tight')
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ###########################################################

    for filename in os.listdir(folder):
        # read one test image
        img = cv2.imread(os.path.join(folder,filename))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #Display results:
        #(1) Segmented images
        mask_img = my_detector.segment_image(img)

        #(2) Stop sign bounding box
        boxes = my_detector.get_bounding_box(img)

        #The autograder checks your answers to the functions segment_image() and get_bounding_box()
        #Make sure your code runs as expected on the testset before submitting to Gradescope
