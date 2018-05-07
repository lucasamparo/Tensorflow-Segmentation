#!/usr/bin/env Rscript
pdf(file="roc.pdf")

par(mgp = c(2.2,0.45,0), tcl = -0.4, mar = c(3.3,3.3,3.3,1.1))
plot(c(0:1.0), c(0:1.0), xlab = "FAR", ylab = "1 - FRR", type="n", xlim = range(0:1), ylim = range(0:1.0), xaxs="i", yaxs="i")
grid(col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)
par(new = TRUE)
plot(c(0:1.0), c(0:1.0), xlab = "FAR", ylab = "1 - FRR", type="n", xlim = range(0:1), ylim = range(0:1.0), xaxs="i", yaxs="i")
legend("bottomright", inset=.05, cex = 0.85, c("CNN Raw","CNN Net","Eigen Raw","Eigen Net","Fisher Raw","Fisher Net","LBPH Raw","LBPH Net","HoG Raw","HoG Net", "Random"), lty=c(1,5,1,5,1,5,1,5,1,5,5), lwd=c(2,2,2,2,2,2,2,2,2,2,2), col=c("blue", "blue", "green", "green","red","red","yellow3","yellow3","purple","purple","black"), bg="grey96")

#CNN
far <- scan("far_cnn_expressas.txt")
frr <- scan("frr_cnn_expressas.txt")
par(new = TRUE)
plot(far, 1 - frr, col = "blue", pch = 16, type = "l", xaxs = "i", yaxs = "i", axes = FALSE, ann = FALSE, xlim = range(0:1), ylim = range(0:1.0))

far <- scan("far_cnn_rede.txt")
frr <- scan("frr_cnn_rede.txt")
par(new = TRUE)
plot(far, 1 - frr, col = "blue", lty=5, pch = 16, type = "l", xaxs = "i", yaxs = "i", axes = FALSE, ann = FALSE, xlim = range(0:1), ylim = range(0:1.0))

#Eigen
far <- scan("far_eigen_expressas.txt")
frr <- scan("frr_eigen_expressas.txt")
par(new = TRUE)
plot(far, 1 - frr, col = "green", pch = 16, type = "l", xaxs = "i", yaxs = "i", axes = FALSE, ann = FALSE, xlim = range(0:1), ylim = range(0:1.0))

far <- scan("far_eigen_rede.txt")
frr <- scan("frr_eigen_rede.txt")
par(new = TRUE)
plot(far, 1 - frr, col = "green", lty=5, pch = 16, type = "l", xaxs = "i", yaxs = "i", axes = FALSE, ann = FALSE, xlim = range(0:1), ylim = range(0:1.0))

#Fisher
far <- scan("far_fisher_expressas.txt")
frr <- scan("frr_fisher_expressas.txt")
par(new = TRUE)
plot(far, 1 - frr, col = "red", pch = 16, type = "l", xaxs = "i", yaxs = "i", axes = FALSE, ann = FALSE, xlim = range(0:1), ylim = range(0:1.0))

far <- scan("far_fisher_rede.txt")
frr <- scan("frr_fisher_rede.txt")
par(new = TRUE)
plot(far, 1 - frr, col = "red", lty=5, pch = 16, type = "l", xaxs = "i", yaxs = "i", axes = FALSE, ann = FALSE, xlim = range(0:1), ylim = range(0:1.0))

#LBPH
far <- scan("far_lbph_expressas.txt")
frr <- scan("frr_lbph_expressas.txt")
par(new = TRUE)
plot(far, 1 - frr, col = "yellow3", pch = 16, type = "l", xaxs = "i", yaxs = "i", axes = FALSE, ann = FALSE, xlim = range(0:1), ylim = range(0:1.0))

far <- scan("far_lbph_rede.txt")
frr <- scan("frr_lbph_rede.txt")
par(new = TRUE)
plot(far, 1 - frr, col = "yellow3", lty=5, pch = 16, type = "l", xaxs = "i", yaxs = "i", axes = FALSE, ann = FALSE, xlim = range(0:1), ylim = range(0:1.0))

#HoG
far <- scan("far_hog_expressas.txt")
frr <- scan("frr_hog_expressas.txt")
par(new = TRUE)
plot(far, 1 - frr, col = "purple", pch = 16, type = "l", xaxs = "i", yaxs = "i", axes = FALSE, ann = FALSE, xlim = range(0:1), ylim = range(0:1.0))

far <- scan("far_hog_rede.txt")
frr <- scan("frr_hog_rede.txt")
par(new = TRUE)
plot(far, 1 - frr, col = "purple", lty=5, pch = 16, type = "l", xaxs = "i", yaxs = "i", axes = FALSE, ann = FALSE, xlim = range(0:1), ylim = range(0:1.0))

par(new = TRUE)
plot(c(0:1.0), c(0:1.0), lty = 5, col = "black", pch = 16, type = "l", xaxs = "i", yaxs = "i", axes = FALSE, ann = FALSE, xlim = range(0:1), ylim = range(0:1.0))

dev.off()
