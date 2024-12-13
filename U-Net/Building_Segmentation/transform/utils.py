def crop_image(image, target_image_dims=[1500, 1500, 3]):
	target_size = target_image_dims[0]
	image_size = len(image)
	padding = (image_size - target_size) // 2

	return image[
		padding: image_size - padding,
		padding: image_size - padding,
		:
	]