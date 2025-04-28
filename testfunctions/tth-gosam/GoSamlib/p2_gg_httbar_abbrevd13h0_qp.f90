module     p2_gg_httbar_abbrevd13h0_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh0_qp
   implicit none
   private
   complex(ki), dimension(60), public :: abb13
   complex(ki), public :: R2d13
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_color_qp, only: TR
      use p2_gg_httbar_globalsl1_qp, only: epspow
      implicit none
      abb13(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb13(2)=es12**(-1)
      abb13(3)=spbl4k2**(-1)
      abb13(4)=spak2l3**(-1)
      abb13(5)=spbl3k2**(-1)
      abb13(6)=spbl5k2**(-1)
      abb13(7)=sqrt(mT**2)
      abb13(8)=spak2l4**(-1)
      abb13(9)=c1-c2
      abb13(9)=abb13(9)*spae1e2*NC*gs**4*i_*TR*spbe2e1*e*gHT*abb13(1)
      abb13(10)=abb13(2)*abb13(9)
      abb13(11)=-mT*abb13(10)
      abb13(12)=abb13(11)*abb13(7)
      abb13(13)=abb13(3)*spbk2k1
      abb13(14)=abb13(12)*abb13(13)
      abb13(15)=mT**2
      abb13(16)=-abb13(15)*abb13(10)
      abb13(17)=abb13(16)*abb13(13)
      abb13(18)=abb13(14)+abb13(17)
      abb13(19)=abb13(18)*spak1l5
      abb13(20)=spbl3k2*abb13(6)
      abb13(21)=abb13(17)*abb13(20)
      abb13(22)=abb13(21)*spak1l3
      abb13(22)=abb13(22)+abb13(19)
      abb13(23)=abb13(16)*spbk2k1
      abb13(24)=abb13(7)*spbk2k1
      abb13(25)=abb13(11)*abb13(24)
      abb13(26)=abb13(23)+abb13(25)
      abb13(27)=spak1l4*abb13(6)
      abb13(28)=abb13(26)*abb13(27)
      abb13(29)=mH**2*abb13(5)*abb13(4)
      abb13(30)=abb13(29)*spak2l5
      abb13(31)=abb13(30)*spbk2k1
      abb13(32)=abb13(10)*spak1l4
      abb13(33)=abb13(31)*abb13(32)
      abb13(34)=abb13(10)*spal3l5
      abb13(35)=abb13(34)*spak1l4
      abb13(36)=abb13(35)*spbl3k1
      abb13(36)=abb13(33)+abb13(36)-abb13(28)
      abb13(37)=abb13(34)*spak2l4
      abb13(38)=abb13(37)*spbl3k2
      abb13(39)=-abb13(38)+abb13(36)
      abb13(40)=abb13(39)-abb13(22)
      abb13(41)=1.0_ki/4.0_ki*abb13(40)
      abb13(26)=abb13(26)*abb13(6)
      abb13(42)=spak1l4**2
      abb13(43)=abb13(42)*abb13(26)
      abb13(19)=spak1l4*abb13(19)
      abb13(44)=3.0_ki*spak1l4
      abb13(45)=abb13(38)*abb13(44)
      abb13(19)=abb13(19)+abb13(43)+abb13(45)
      abb13(19)=spbl4k1*abb13(19)
      abb13(43)=-abb13(23)+abb13(25)
      abb13(45)=abb13(20)*spak2l3
      abb13(43)=abb13(43)*abb13(45)
      abb13(46)=abb13(10)*abb13(7)**2
      abb13(47)=-spbk2k1*abb13(46)
      abb13(23)=-abb13(23)+abb13(47)
      abb13(23)=spak2l5*abb13(23)
      abb13(23)=abb13(43)+abb13(23)
      abb13(23)=abb13(44)*abb13(23)
      abb13(43)=mT**4
      abb13(44)=-abb13(43)*abb13(9)
      abb13(47)=abb13(44)*abb13(13)
      abb13(48)=mT**3
      abb13(49)=-abb13(48)*abb13(9)
      abb13(50)=abb13(13)*abb13(7)
      abb13(51)=abb13(49)*abb13(50)
      abb13(47)=abb13(47)+abb13(51)
      abb13(47)=abb13(8)*abb13(47)*abb13(27)
      abb13(51)=abb13(3)**2
      abb13(52)=abb13(44)*abb13(51)
      abb13(53)=abb13(52)*spbk2k1
      abb13(54)=abb13(49)*abb13(51)
      abb13(55)=abb13(24)*abb13(54)
      abb13(53)=abb13(53)+abb13(55)
      abb13(53)=abb13(8)*abb13(53)
      abb13(55)=-abb13(10)*abb13(24)
      abb13(56)=spbk2k1*abb13(11)
      abb13(55)=abb13(56)+abb13(55)
      abb13(56)=3.0_ki*spak2l4
      abb13(55)=abb13(7)*abb13(55)*abb13(56)
      abb13(53)=abb13(55)+abb13(53)
      abb13(53)=spak1l5*abb13(53)
      abb13(15)=-abb13(15)*abb13(9)
      abb13(55)=abb13(8)*abb13(15)*spak2l5
      abb13(57)=spak1l4*abb13(13)*abb13(55)
      abb13(42)=abb13(42)*spbl4k1
      abb13(58)=abb13(10)*spak2l5
      abb13(59)=-spbk2k1*abb13(58)*abb13(42)
      abb13(57)=abb13(57)+abb13(59)
      abb13(57)=abb13(57)*abb13(29)
      abb13(59)=abb13(8)*spal3l5
      abb13(15)=abb13(3)*abb13(59)*abb13(15)*spak1l4
      abb13(42)=-abb13(34)*abb13(42)
      abb13(15)=abb13(15)+abb13(42)
      abb13(15)=spbl3k1*abb13(15)
      abb13(28)=3.0_ki*abb13(28)+abb13(38)
      abb13(28)=spak2l4*abb13(28)
      abb13(33)=-abb13(33)*abb13(56)
      abb13(42)=3.0_ki*abb13(37)
      abb13(60)=-abb13(42)*spbl3k1*spak1l4
      abb13(28)=abb13(60)+abb13(28)+abb13(33)
      abb13(28)=spbl4k2*abb13(28)
      abb13(25)=abb13(25)*abb13(56)
      abb13(33)=abb13(52)*abb13(8)
      abb13(56)=spbk2k1*abb13(33)
      abb13(25)=abb13(25)+abb13(56)
      abb13(25)=abb13(20)*abb13(25)
      abb13(56)=spbl4k1*spak1l4*abb13(21)
      abb13(25)=abb13(56)+abb13(25)
      abb13(25)=spak1l3*abb13(25)
      abb13(56)=spal3l5*spbl3k2
      abb13(9)=abb13(7)*abb13(3)*abb13(56)*mT*abb13(9)
      abb13(9)=abb13(25)+abb13(28)+abb13(15)+abb13(57)+abb13(19)+abb13(53)+3.0_&
      &ki*abb13(9)+abb13(47)+abb13(23)
      abb13(9)=1.0_ki/4.0_ki*abb13(9)
      abb13(15)=-1.0_ki/2.0_ki*abb13(39)
      abb13(19)=-abb13(43)*abb13(10)
      abb13(13)=abb13(19)*abb13(13)
      abb13(23)=abb13(10)*abb13(48)
      abb13(25)=-abb13(23)*abb13(50)
      abb13(13)=abb13(13)+abb13(25)
      abb13(13)=abb13(13)*abb13(27)
      abb13(23)=-abb13(23)*abb13(51)*abb13(24)
      abb13(19)=abb13(19)*abb13(51)*spbk2k1
      abb13(23)=abb13(19)+abb13(23)
      abb13(23)=spak1l5*abb13(23)
      abb13(24)=spak1l4*abb13(17)*abb13(30)
      abb13(19)=spak1l3*abb13(20)*abb13(19)
      abb13(13)=abb13(13)+abb13(23)+abb13(24)+abb13(19)
      abb13(13)=1.0_ki/2.0_ki*abb13(13)
      abb13(13)=abb13(8)*abb13(13)
      abb13(19)=abb13(10)*abb13(7)
      abb13(23)=-abb13(11)+2.0_ki*abb13(19)
      abb13(23)=abb13(23)*abb13(7)
      abb13(23)=abb13(23)+abb13(16)
      abb13(23)=abb13(23)*spal4l5
      abb13(24)=-abb13(16)+2.0_ki*abb13(12)
      abb13(24)=abb13(24)*abb13(20)*spal3l4
      abb13(23)=abb13(23)+abb13(24)
      abb13(24)=abb13(16)*abb13(3)
      abb13(25)=abb13(12)*abb13(3)
      abb13(28)=-abb13(24)+3.0_ki/2.0_ki*abb13(25)
      abb13(28)=abb13(28)*abb13(56)
      abb13(13)=abb13(28)+1.0_ki/2.0_ki*abb13(38)+abb13(13)+abb13(23)
      abb13(22)=abb13(36)-abb13(22)
      abb13(28)=abb13(24)-2.0_ki*abb13(25)
      abb13(28)=abb13(28)*abb13(56)
      abb13(22)=abb13(28)-abb13(23)-1.0_ki/2.0_ki*abb13(22)
      abb13(23)=-1.0_ki/2.0_ki*abb13(40)
      abb13(28)=1.0_ki/4.0_ki*spak1l4
      abb13(36)=abb13(18)*abb13(28)
      abb13(38)=abb13(7)*abb13(54)
      abb13(38)=abb13(52)+abb13(38)
      abb13(38)=abb13(8)*abb13(38)
      abb13(11)=abb13(19)-abb13(11)
      abb13(11)=abb13(11)*abb13(7)
      abb13(19)=spak2l4*abb13(11)
      abb13(19)=abb13(19)+abb13(38)
      abb13(19)=1.0_ki/4.0_ki*abb13(19)
      abb13(38)=abb13(25)+abb13(24)
      abb13(39)=-abb13(28)*abb13(11)
      abb13(40)=-spbl4k1*abb13(35)
      abb13(37)=spbl4k2*abb13(37)
      abb13(37)=abb13(40)+abb13(37)
      abb13(37)=1.0_ki/4.0_ki*abb13(37)
      abb13(40)=abb13(12)+abb13(16)
      abb13(27)=abb13(40)*abb13(27)
      abb13(43)=spak1l5*abb13(38)
      abb13(47)=abb13(20)*abb13(24)
      abb13(48)=spak1l3*abb13(47)
      abb13(27)=abb13(48)+abb13(27)+abb13(43)
      abb13(27)=spbl4k1*abb13(27)
      abb13(43)=abb13(7)*abb13(49)
      abb13(43)=abb13(44)+abb13(43)
      abb13(43)=abb13(8)*abb13(6)*abb13(3)*abb13(43)
      abb13(40)=abb13(40)*abb13(6)
      abb13(44)=-spak2l4*abb13(40)
      abb13(10)=abb13(10)*spak2l4
      abb13(30)=abb13(10)*abb13(30)
      abb13(30)=abb13(44)+abb13(30)
      abb13(30)=spbl4k2*abb13(30)
      abb13(44)=-abb13(46)-abb13(16)
      abb13(44)=spak2l5*abb13(44)
      abb13(46)=abb13(3)*abb13(55)
      abb13(32)=-spbl4k1*spak2l5*abb13(32)
      abb13(32)=abb13(46)+abb13(32)
      abb13(32)=abb13(32)*abb13(29)
      abb13(16)=abb13(12)-abb13(16)
      abb13(16)=abb13(16)*abb13(45)
      abb13(24)=spak1k2*spbl3k1*abb13(24)*abb13(59)
      abb13(16)=abb13(24)+abb13(16)+abb13(30)+abb13(32)+abb13(43)+abb13(44)+abb&
      &13(27)
      abb13(16)=1.0_ki/4.0_ki*abb13(16)
      abb13(24)=abb13(58)*abb13(29)
      abb13(24)=abb13(24)-abb13(40)
      abb13(11)=spak1l5*abb13(11)
      abb13(12)=abb13(12)*abb13(20)
      abb13(27)=-spak1l3*abb13(12)
      abb13(11)=abb13(11)+abb13(27)
      abb13(11)=1.0_ki/4.0_ki*abb13(11)
      abb13(27)=abb13(21)*abb13(28)
      abb13(29)=-spak2l4*abb13(12)
      abb13(20)=abb13(20)*abb13(33)
      abb13(20)=abb13(29)+abb13(20)
      abb13(20)=1.0_ki/4.0_ki*abb13(20)
      abb13(12)=abb13(28)*abb13(12)
      abb13(17)=abb13(17)*abb13(28)*abb13(59)
      abb13(28)=3.0_ki/4.0_ki*spbl3k2*abb13(35)
      abb13(14)=-1.0_ki/4.0_ki*spal3l5*abb13(14)
      abb13(26)=spak2l4*abb13(26)
      abb13(18)=-spak2l5*abb13(18)
      abb13(10)=-abb13(10)*abb13(31)
      abb13(21)=spak2l3*abb13(21)
      abb13(10)=-abb13(21)+abb13(10)+abb13(26)+abb13(18)
      abb13(18)=spal3l5*abb13(25)
      abb13(18)=abb13(18)-abb13(42)
      abb13(18)=spbl3k1*abb13(18)
      abb13(10)=abb13(18)+3.0_ki*abb13(10)
      abb13(10)=1.0_ki/4.0_ki*abb13(10)
      R2d13=abb13(41)
      rat2 = rat2 + R2d13
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='13' value='", &
          & R2d13, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd13h0_qp
