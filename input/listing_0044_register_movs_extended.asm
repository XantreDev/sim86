; I've implemented mov-s for 8 bit registers, so let's test it

bits 16

mov ax, 0x2222
mov bx, 0x4444
mov cx, 0x6666
mov dx, 0x8888

mov al, 0x11
mov bh, 0x33
mov cl, 0x55
mov dh, 0x77

mov ah, bl
mov cl, dh
