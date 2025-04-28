module     p2_gg_httbar_abbrevd11h12_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh12_qp
   implicit none
   private
   complex(ki), dimension(60), public :: abb11
   complex(ki), public :: R2d11
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
      abb11(1)=1.0_ki/(-mT**2+es34)
      abb11(2)=es12**(-1)
      abb11(3)=spak2l3**(-1)
      abb11(4)=spbl3k2**(-1)
      abb11(5)=spak2l4**(-1)
      abb11(6)=spak2l5**(-1)
      abb11(7)=sqrt(mT**2)
      abb11(8)=spbl5k2**(-1)
      abb11(9)=c1-c2
      abb11(9)=abb11(9)*spae1e2*NC*gs**4*i_*TR*spbe2e1*e*gHT*abb11(1)
      abb11(10)=abb11(2)*abb11(9)
      abb11(11)=-mT*abb11(10)
      abb11(12)=abb11(11)*abb11(7)
      abb11(13)=abb11(6)*spak1k2
      abb11(14)=abb11(12)*abb11(13)
      abb11(15)=mT**2
      abb11(16)=-abb11(15)*abb11(10)
      abb11(17)=abb11(16)*abb11(13)
      abb11(18)=abb11(14)+abb11(17)
      abb11(19)=abb11(18)*spbl4k1
      abb11(20)=spak2l3*abb11(5)
      abb11(21)=abb11(17)*abb11(20)
      abb11(22)=abb11(21)*spbl3k1
      abb11(22)=abb11(22)+abb11(19)
      abb11(23)=abb11(16)*spak1k2
      abb11(24)=abb11(7)*spak1k2
      abb11(25)=abb11(11)*abb11(24)
      abb11(26)=abb11(23)+abb11(25)
      abb11(27)=spbl5k1*abb11(5)
      abb11(28)=abb11(26)*abb11(27)
      abb11(29)=mH**2*abb11(4)*abb11(3)
      abb11(30)=abb11(29)*spbl4k2
      abb11(31)=abb11(30)*spak1k2
      abb11(32)=abb11(10)*spbl5k1
      abb11(33)=abb11(31)*abb11(32)
      abb11(34)=abb11(10)*spbl4l3
      abb11(35)=abb11(34)*spbl5k1
      abb11(36)=abb11(35)*spak1l3
      abb11(36)=abb11(33)+abb11(36)-abb11(28)
      abb11(37)=abb11(34)*spbl5k2
      abb11(38)=abb11(37)*spak2l3
      abb11(39)=-abb11(38)+abb11(36)
      abb11(40)=abb11(39)-abb11(22)
      abb11(41)=-1.0_ki/4.0_ki*abb11(40)
      abb11(26)=abb11(26)*abb11(5)
      abb11(42)=spbl5k1**2
      abb11(43)=-abb11(42)*abb11(26)
      abb11(19)=-spbl5k1*abb11(19)
      abb11(44)=3.0_ki*spbl5k1
      abb11(45)=-abb11(38)*abb11(44)
      abb11(19)=abb11(19)+abb11(43)+abb11(45)
      abb11(19)=spak1l5*abb11(19)
      abb11(43)=abb11(23)-abb11(25)
      abb11(45)=abb11(20)*spbl3k2
      abb11(43)=abb11(43)*abb11(45)
      abb11(46)=abb11(10)*abb11(7)**2
      abb11(47)=spak1k2*abb11(46)
      abb11(23)=abb11(23)+abb11(47)
      abb11(23)=spbl4k2*abb11(23)
      abb11(23)=abb11(43)+abb11(23)
      abb11(23)=abb11(44)*abb11(23)
      abb11(43)=mT**4
      abb11(44)=-abb11(43)*abb11(9)
      abb11(47)=-abb11(44)*abb11(13)
      abb11(48)=mT**3
      abb11(49)=-abb11(48)*abb11(9)
      abb11(50)=abb11(13)*abb11(7)
      abb11(51)=-abb11(49)*abb11(50)
      abb11(47)=abb11(47)+abb11(51)
      abb11(47)=abb11(8)*abb11(47)*abb11(27)
      abb11(51)=abb11(6)**2
      abb11(52)=abb11(44)*abb11(51)
      abb11(53)=abb11(52)*spak1k2
      abb11(54)=abb11(49)*abb11(51)
      abb11(55)=-abb11(24)*abb11(54)
      abb11(53)=-abb11(53)+abb11(55)
      abb11(53)=abb11(8)*abb11(53)
      abb11(55)=abb11(10)*abb11(24)
      abb11(56)=-spak1k2*abb11(11)
      abb11(55)=abb11(56)+abb11(55)
      abb11(56)=3.0_ki*spbl5k2
      abb11(55)=abb11(7)*abb11(55)*abb11(56)
      abb11(53)=abb11(55)+abb11(53)
      abb11(53)=spbl4k1*abb11(53)
      abb11(15)=-abb11(15)*abb11(9)
      abb11(55)=abb11(8)*abb11(15)*spbl4k2
      abb11(57)=-spbl5k1*abb11(13)*abb11(55)
      abb11(42)=abb11(42)*spak1l5
      abb11(58)=abb11(10)*spbl4k2
      abb11(59)=spak1k2*abb11(58)*abb11(42)
      abb11(57)=abb11(57)+abb11(59)
      abb11(57)=abb11(57)*abb11(29)
      abb11(59)=abb11(8)*spbl4l3
      abb11(15)=-abb11(6)*abb11(59)*abb11(15)*spbl5k1
      abb11(42)=abb11(34)*abb11(42)
      abb11(15)=abb11(15)+abb11(42)
      abb11(15)=spak1l3*abb11(15)
      abb11(28)=-3.0_ki*abb11(28)-abb11(38)
      abb11(28)=spbl5k2*abb11(28)
      abb11(33)=abb11(33)*abb11(56)
      abb11(42)=3.0_ki*abb11(37)
      abb11(60)=abb11(42)*spak1l3*spbl5k1
      abb11(28)=abb11(60)+abb11(28)+abb11(33)
      abb11(28)=spak2l5*abb11(28)
      abb11(25)=-abb11(25)*abb11(56)
      abb11(33)=abb11(52)*abb11(8)
      abb11(56)=-spak1k2*abb11(33)
      abb11(25)=abb11(25)+abb11(56)
      abb11(25)=abb11(20)*abb11(25)
      abb11(56)=-spak1l5*spbl5k1*abb11(21)
      abb11(25)=abb11(56)+abb11(25)
      abb11(25)=spbl3k1*abb11(25)
      abb11(56)=spbl4l3*spak2l3
      abb11(9)=-abb11(7)*abb11(6)*abb11(56)*mT*abb11(9)
      abb11(9)=abb11(25)+abb11(28)+abb11(15)+abb11(57)+abb11(19)+abb11(53)+3.0_&
      &ki*abb11(9)+abb11(47)+abb11(23)
      abb11(9)=1.0_ki/4.0_ki*abb11(9)
      abb11(15)=1.0_ki/2.0_ki*abb11(39)
      abb11(19)=-abb11(43)*abb11(10)
      abb11(13)=-abb11(19)*abb11(13)
      abb11(23)=abb11(10)*abb11(48)
      abb11(25)=abb11(23)*abb11(50)
      abb11(13)=abb11(13)+abb11(25)
      abb11(13)=abb11(13)*abb11(27)
      abb11(23)=abb11(23)*abb11(51)*abb11(24)
      abb11(19)=abb11(19)*abb11(51)*spak1k2
      abb11(23)=-abb11(19)+abb11(23)
      abb11(23)=spbl4k1*abb11(23)
      abb11(24)=spbl5k1*abb11(17)*abb11(30)
      abb11(19)=spbl3k1*abb11(20)*abb11(19)
      abb11(13)=-abb11(13)-abb11(23)+abb11(24)+abb11(19)
      abb11(13)=-1.0_ki/2.0_ki*abb11(13)
      abb11(13)=abb11(8)*abb11(13)
      abb11(19)=abb11(10)*abb11(7)
      abb11(23)=-abb11(11)+2.0_ki*abb11(19)
      abb11(23)=abb11(23)*abb11(7)
      abb11(23)=abb11(23)+abb11(16)
      abb11(23)=abb11(23)*spbl5l4
      abb11(24)=-abb11(16)+2.0_ki*abb11(12)
      abb11(24)=abb11(24)*abb11(20)*spbl5l3
      abb11(23)=abb11(23)-abb11(24)
      abb11(24)=abb11(16)*abb11(6)
      abb11(25)=abb11(12)*abb11(6)
      abb11(28)=abb11(24)-3.0_ki/2.0_ki*abb11(25)
      abb11(28)=abb11(28)*abb11(56)
      abb11(13)=abb11(28)-1.0_ki/2.0_ki*abb11(38)+abb11(13)+abb11(23)
      abb11(22)=abb11(36)-abb11(22)
      abb11(28)=-abb11(24)+2.0_ki*abb11(25)
      abb11(28)=abb11(28)*abb11(56)
      abb11(22)=abb11(28)-abb11(23)+1.0_ki/2.0_ki*abb11(22)
      abb11(23)=1.0_ki/2.0_ki*abb11(40)
      abb11(28)=1.0_ki/4.0_ki*spbl5k1
      abb11(36)=-abb11(18)*abb11(28)
      abb11(38)=-abb11(21)*abb11(28)
      abb11(39)=spak1l5*abb11(35)
      abb11(37)=-spak2l5*abb11(37)
      abb11(37)=abb11(39)+abb11(37)
      abb11(37)=1.0_ki/4.0_ki*abb11(37)
      abb11(17)=-abb11(17)*abb11(28)*abb11(59)
      abb11(14)=1.0_ki/4.0_ki*spbl4l3*abb11(14)
      abb11(39)=abb11(12)+abb11(16)
      abb11(27)=-abb11(39)*abb11(27)
      abb11(40)=abb11(25)+abb11(24)
      abb11(43)=-spbl4k1*abb11(40)
      abb11(47)=abb11(20)*abb11(24)
      abb11(48)=-spbl3k1*abb11(47)
      abb11(27)=abb11(48)+abb11(27)+abb11(43)
      abb11(27)=spak1l5*abb11(27)
      abb11(43)=-abb11(7)*abb11(49)
      abb11(43)=-abb11(44)+abb11(43)
      abb11(43)=abb11(8)*abb11(5)*abb11(6)*abb11(43)
      abb11(39)=abb11(39)*abb11(5)
      abb11(44)=spbl5k2*abb11(39)
      abb11(10)=abb11(10)*spbl5k2
      abb11(30)=-abb11(10)*abb11(30)
      abb11(30)=abb11(44)+abb11(30)
      abb11(30)=spak2l5*abb11(30)
      abb11(44)=abb11(46)+abb11(16)
      abb11(44)=spbl4k2*abb11(44)
      abb11(46)=-abb11(6)*abb11(55)
      abb11(32)=spak1l5*spbl4k2*abb11(32)
      abb11(32)=abb11(46)+abb11(32)
      abb11(32)=abb11(32)*abb11(29)
      abb11(16)=-abb11(12)+abb11(16)
      abb11(16)=abb11(16)*abb11(45)
      abb11(24)=-spbk2k1*spak1l3*abb11(24)*abb11(59)
      abb11(16)=abb11(24)+abb11(16)+abb11(30)+abb11(32)+abb11(43)+abb11(44)+abb&
      &11(27)
      abb11(16)=1.0_ki/4.0_ki*abb11(16)
      abb11(24)=abb11(58)*abb11(29)
      abb11(24)=abb11(24)-abb11(39)
      abb11(27)=-abb11(7)*abb11(54)
      abb11(27)=-abb11(52)+abb11(27)
      abb11(27)=abb11(8)*abb11(27)
      abb11(11)=abb11(19)-abb11(11)
      abb11(11)=abb11(11)*abb11(7)
      abb11(19)=-spbl5k2*abb11(11)
      abb11(19)=abb11(19)+abb11(27)
      abb11(19)=1.0_ki/4.0_ki*abb11(19)
      abb11(12)=abb11(12)*abb11(20)
      abb11(27)=spbl5k2*abb11(12)
      abb11(20)=-abb11(20)*abb11(33)
      abb11(20)=abb11(27)+abb11(20)
      abb11(20)=1.0_ki/4.0_ki*abb11(20)
      abb11(26)=-spbl5k2*abb11(26)
      abb11(18)=spbl4k2*abb11(18)
      abb11(10)=abb11(10)*abb11(31)
      abb11(21)=spbl3k2*abb11(21)
      abb11(10)=abb11(21)+abb11(10)+abb11(26)+abb11(18)
      abb11(18)=-spbl4l3*abb11(25)
      abb11(18)=abb11(18)+abb11(42)
      abb11(18)=spak1l3*abb11(18)
      abb11(10)=abb11(18)+3.0_ki*abb11(10)
      abb11(10)=1.0_ki/4.0_ki*abb11(10)
      abb11(18)=-spbl4k1*abb11(11)
      abb11(21)=spbl3k1*abb11(12)
      abb11(18)=abb11(18)+abb11(21)
      abb11(18)=1.0_ki/4.0_ki*abb11(18)
      abb11(11)=abb11(28)*abb11(11)
      abb11(12)=-abb11(28)*abb11(12)
      abb11(21)=-3.0_ki/4.0_ki*spak2l3*abb11(35)
      R2d11=abb11(41)
      rat2 = rat2 + R2d11
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='11' value='", &
          & R2d11, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd11h12_qp
