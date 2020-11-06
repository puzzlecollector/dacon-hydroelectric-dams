def build_unet(start_neurons,k1,k2): 
    inputs = Input((120,120,4)) 
    bn = BatchNormalization()(inputs) 
    
    # compressing layer 
    conv0 = Conv2D(start_neurons * 1, (k1, k1), activation='relu', padding='same', kernel_initializer='he_normal')(bn) 
    conv1 = Conv2D(start_neurons * 1, (k2, k2), activation='relu', padding='same', kernel_initializer='he_normal')(bn) 
    conv11 = concatenate([conv0,conv1]) 
    bn = BatchNormalization()(conv11) 
    maxpool = MaxPooling2D((2, 2))(bn)
    avgpool = AveragePooling2D((2, 2))(bn) 
    pools = concatenate([maxpool,avgpool]) 

    conv0 = Conv2D(start_neurons * 2, (k1, k1), activation='relu', padding='same', kernel_initializer='he_normal')(pools) 
    conv1 = Conv2D(start_neurons * 2, (k2, k2), activation='relu', padding='same', kernel_initializer='he_normal')(pools)
    conv22 = concatenate([conv0,conv1]) 
    bn = BatchNormalization()(conv22)  
    maxpool = MaxPooling2D((2, 2))(bn)
    avgpool = AveragePooling2D((2, 2))(bn) 
    pools = concatenate([maxpool,avgpool]) 

    # middle layer 
    convm0 = Conv2D(start_neurons * 4, (k1, k1), activation='relu', padding='same', kernel_initializer='he_normal')(pools)
    convm1 = Conv2D(start_neurons * 4, (k2, k2), activation='relu', padding='same', kernel_initializer='he_normal')(pools)
    convm = concatenate([convm0,convm1]) 
    bn = BatchNormalization()(convm) 

    # decompressing layer 
    deconv0 = Conv2DTranspose(start_neurons * 2, (k1, k1), strides=(2, 2), padding='same', kernel_initializer='he_normal')(bn)
    deconv1 = Conv2DTranspose(start_neurons * 2, (k2, k2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(bn)
    deconv22 = concatenate([deconv0,deconv1]) 
    uconv2 = concatenate([deconv22, conv22]) 
    uconv20 = Conv2D(start_neurons * 2, (k1, k2), activation='relu', padding='same', kernel_initializer='he_normal')(uconv2)
    uconv21 = Conv2D(start_neurons * 2, (k2, k2), activation='relu', padding='same', kernel_initializer='he_normal')(uconv2) 
    uconv22 = concatenate([uconv20,uconv21]) 
    bn = BatchNormalization()(uconv22)

    deconv0 = Conv2DTranspose(start_neurons * 1, (k1, k1), strides=(2, 2), padding='same', kernel_initializer='he_normal')(bn) 
    deconv1 = Conv2DTranspose(start_neurons * 1, (k2, k2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(bn)
    deconv11 = concatenate([deconv0,deconv1]) 
    uconv1 = concatenate([deconv11,conv11]) 
    uconv10 = Conv2D(start_neurons * 1, (k1, k1), activation = 'relu', padding='same', kernel_initializer='he_normal')(uconv1)
    uconv11 = Conv2D(start_neurons * 1, (k2, k2), activation = 'relu', padding='same', kernel_initializer='he_normal')(uconv1) 
    uconv11 = concatenate([uconv10,uconv11]) 
    bn = BatchNormalization()(uconv11) 
    outputs = Conv2D(1, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(bn)
    model = Model(inputs=inputs,outputs=outputs) 
    model.compile(loss='mae',optimizer='adam',metrics=['mae']) 
    return model 
