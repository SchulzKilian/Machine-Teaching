def stop_for_validation(validation_losses):
    if len(validation_losses) < 4:
        return False
    else:
        y1, y2 = sum(validation_losses[-4:-2]) / 2, sum(validation_losses[-2:]) / 2
        slope = (y2 - y1) / 2 
        condition = slope <= 0
        if condition:
            print("Validation loss stop condition triggered")
            return True
        else:
            return False


def stop_for_pixel_loss(positives, negatives):
    if len(positives) < 3 or len(negatives) < 3:
        return False
    else:
        pos_slope = (positives[-1] - positives[-3]) / 2
        neg_slope = (negatives[-1] - negatives[-3]) / 2 * -1
        condition = pos_slope + neg_slope <= 0
        if condition:
            print("Pixel loss stop condition triggered")
            return True
        return False