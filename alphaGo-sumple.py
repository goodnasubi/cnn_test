#from chainer import cuda, Variable, FunctionSet, optimizers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class CaiwaGo():
    def __init__(self, train=True):
        super(CaiwaGo,self).__init__(
            conv0 = L.Convolution2D(17,256,3, pad=1),
            conv1 = L.Convolution2D(256,256,3, pad=1),
            conv2 = L.Convolution2D(256,256,3, pad=1),
            conv3 = L.Convolution2D(256,256,3, pad=1),
            conv4 = L.Convolution2D(256,256,3, pad=1),
            conv5 = L.Convolution2D(256,256,3, pad=1),
            conv6 = L.Convolution2D(256,256,3, pad=1),
            conv7 = L.Convolution2D(256,256,3, pad=1),
            conv8 = L.Convolution2D(256,256,3, pad=1),
            conv9 = L.Convolution2D(256,256,3, pad=1),
            conv10 = L.Convolution2D(256,256,3, pad=1),
            conv11 = L.Convolution2D(256,256,3, pad=1),
            conv12 = L.Convolution2D(256,256,3, pad=1),
            conv13 = L.Convolution2D(256,256,3, pad=1),
            conv14 = L.Convolution2D(256,256,3, pad=1),
            conv15 = L.Convolution2D(256,256,3, pad=1),
            conv16 = L.Convolution2D(256,256,3, pad=1),
            conv17 = L.Convolution2D(256,256,3, pad=1),
            conv18 = L.Convolution2D(256,256,3, pad=1),
            conv19 = L.Convolution2D(256,256,3, pad=1),
            conv20 = L.Convolution2D(256,256,3, pad=1),
            conv21 = L.Convolution2D(256,256,3, pad=1),
            conv22 = L.Convolution2D(256,256,3, pad=1),
            conv23 = L.Convolution2D(256,256,3, pad=1),
            conv24 = L.Convolution2D(256,256,3, pad=1),
            conv25 = L.Convolution2D(256,256,3, pad=1),
            conv26 = L.Convolution2D(256,256,3, pad=1),
            conv27 = L.Convolution2D(256,256,3, pad=1),
            conv28 = L.Convolution2D(256,256,3, pad=1),
            conv29 = L.Convolution2D(256,256,3, pad=1),
            conv30 = L.Convolution2D(256,256,3, pad=1),
            conv31 = L.Convolution2D(256,256,3, pad=1),
            conv32 = L.Convolution2D(256,256,3, pad=1),
            conv33 = L.Convolution2D(256,256,3, pad=1),
            conv34 = L.Convolution2D(256,256,3, pad=1),
            conv35 = L.Convolution2D(256,256,3, pad=1),
            conv36 = L.Convolution2D(256,256,3, pad=1),
            conv37 = L.Convolution2D(256,256,3, pad=1),
            conv38 = L.Convolution2D(256,256,3, pad=1),

            bn0 = L.BatchNormalization(256),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(256),
            bn4 = L.BatchNormalization(256),
            bn5 = L.BatchNormalization(256),
            bn6 = L.BatchNormalization(256),
            bn7 = L.BatchNormalization(256),
            bn8 = L.BatchNormalization(256),
            bn9 = L.BatchNormalization(256),
            bn10 = L.BatchNormalization(256),
            bn11 = L.BatchNormalization(256),
            bn12 = L.BatchNormalization(256),
            bn13 = L.BatchNormalization(256),
            bn14 = L.BatchNormalization(256),
            bn15 = L.BatchNormalization(256),
            bn16 = L.BatchNormalization(256),
            bn17 = L.BatchNormalization(256),
            bn18 = L.BatchNormalization(256),
            bn19 = L.BatchNormalization(256),
            bn20 = L.BatchNormalization(256),
            bn21 = L.BatchNormalization(256),
            bn22 = L.BatchNormalization(256),
            bn23 = L.BatchNormalization(256),
            bn24 = L.BatchNormalization(256),
            bn25 = L.BatchNormalization(256),
            bn26 = L.BatchNormalization(256),
            bn27 = L.BatchNormalization(256),
            bn28 = L.BatchNormalization(256),
            bn29 = L.BatchNormalization(256),
            bn30 = L.BatchNormalization(256),
            bn31 = L.BatchNormalization(256),
            bn32 = L.BatchNormalization(256),
            bn33 = L.BatchNormalization(256),
            bn34 = L.BatchNormalization(256),
            bn35 = L.BatchNormalization(256),
            bn36 = L.BatchNormalization(256),
            bn37 = L.BatchNormalization(256),
            bn38 = L.BatchNormalization(256),

            conv_p1 = L.Convolution2D(256,2,1),
            bn_p1 = L.BatchNormalization(2),
            fc_p2 = L.Linear(19*19*2, 19*19),

            conv_v1 = L.Convolution2D(256,1,1),
            bn_v1 = L.BatchNormalization(1),
            fc_v2 = L.Linear(19*19, 256),
            fc_v3 = L.Linear(256, 1),

        )
        
    def __call__(self, x):
        h0 = F.relu(self.bn0(self.conv0(x)))

        h1 = F.relu(self.bn1(self.conv1(h0)))
        h2 = F.relu(self.bn2(self.conv2(h1)) + h0)

        h3 = F.relu(self.bn3(self.conv3(h2)))
        h4 = F.relu(self.bn4(self.conv4(h3)) + h2)

        h5 = F.relu(self.bn5(self.conv5(h4)))
        h6 = F.relu(self.bn6(self.conv6(h5)) + h4)

        h7 = F.relu(self.bn7(self.conv7(h6)))
        h8 = F.relu(self.bn6(self.conv8(h7)) + h6)

        h9 = F.relu(self.bn9(self.conv9(h8)))
        h10 = F.relu(self.bn10(self.conv10(h9)) + h8)

        h9 = F.relu(self.bn9(self.conv9(h8)))
        h10 = F.relu(self.bn10(self.conv10(h9)) + h8)

        h11 = F.relu(self.bn11(self.conv11(h10)))
        h12 = F.relu(self.bn12(self.conv12(h11)) + h10)

        h13 = F.relu(self.bn13(self.conv13(h12)))
        h14 = F.relu(self.bn14(self.conv14(h13)) + h12)

        h15 = F.relu(self.bn15(self.conv15(h14)))
        h16 = F.relu(self.bn16(self.conv16(h15)) + h14)

        h17 = F.relu(self.bn17(self.conv17(h16)))
        h18 = F.relu(self.bn18(self.conv18(h17)) + h16)

        h19 = F.relu(self.bn19(self.conv19(h18)))
        h20 = F.relu(self.bn20(self.conv20(h19)) + h18)

        h21 = F.relu(self.bn21(self.conv20(h20)))
        h22 = F.relu(self.bn22(self.conv22(h21)) + h20)

        h23 = F.relu(self.bn23(self.conv23(h22)))
        h24 = F.relu(self.bn24(self.conv24(h23)) + h22)

        h25 = F.relu(self.bn25(self.conv25(h24)))
        h26 = F.relu(self.bn26(self.conv26(h25)) + h24)

        h27 = F.relu(self.bn27(self.conv27(h26)))
        h28 = F.relu(self.bn28(self.conv28(h27)) + h26)

        h29 = F.relu(self.bn29(self.conv29(h28)))
        h30 = F.relu(self.bn30(self.conv30(h29)) + h28)

        h31 = F.relu(self.bn31(self.conv31(h30)))
        h32 = F.relu(self.bn32(self.conv32(h31)) + h30)

        h33 = F.relu(self.bn33(self.conv33(h32)))
        h34 = F.relu(self.bn34(self.conv34(h33)) + h32)

        h35 = F.relu(self.bn35(self.conv35(h34)))
        h36 = F.relu(self.bn36(self.conv36(h35)) + h34)

        h37 = F.relu(self.bn37(self.conv37(h36)))
        h38 = F.relu(self.bn38(self.conv38(h37)) + h36)

        # policy output
        h_p1 = F.relu(self.bn_p1(self.conv_p1(h38)))
        out_p = self.fc_p2(h_p1)

        # value outout
        h_v1  = F.relu(self.bn_v1(self.conv_v1(h38)))
        h_v2  = F.relu(self.fc_v2(h_v1))
        out_v = F.tanh(self.fc_v3(h_v2))

        return out_p, out_v
        

