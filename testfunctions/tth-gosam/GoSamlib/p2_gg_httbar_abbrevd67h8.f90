module     p2_gg_httbar_abbrevd67h8
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh8
   implicit none
   private
   complex(ki), dimension(54), public :: abb67
   complex(ki), public :: R2d67
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_model
      use p2_gg_httbar_color, only: TR
      use p2_gg_httbar_globalsl1, only: epspow
      implicit none
      abb67(1)=sqrt(mT**2)
      abb67(2)=NC**(-1)
      abb67(3)=es12**(-1)
      abb67(4)=spbl4k2**(-1)
      abb67(5)=spak2l5**(-1)
      abb67(6)=spbl5k2**(-1)
      abb67(7)=spak2l3**(-1)
      abb67(8)=spbl3k2**(-1)
      abb67(9)=spak2l4**(-1)
      abb67(10)=spak1l4*spbl4k1
      abb67(11)=spak2l4*spbl4k2
      abb67(12)=spak2l3*spbl3k2
      abb67(10)=abb67(12)+abb67(10)-abb67(11)
      abb67(10)=abb67(10)*spak2l4
      abb67(11)=spbl3k1*spak1k2
      abb67(13)=abb67(11)*spal3l4
      abb67(14)=spak1l4*spbl3k1
      abb67(15)=abb67(14)*spak2l3
      abb67(16)=spbl5k1*spak1k2
      abb67(17)=abb67(16)*spal4l5
      abb67(10)=abb67(10)-abb67(17)-abb67(13)-abb67(15)
      abb67(10)=abb67(10)*abb67(5)
      abb67(13)=spbl5k1*spak1l5
      abb67(18)=spbl5k2*spak2l5
      abb67(13)=abb67(12)+abb67(13)-abb67(18)
      abb67(13)=abb67(13)*spbl5k2
      abb67(18)=spak1l4*spbk2k1
      abb67(19)=abb67(18)*spbl5l4
      abb67(20)=spbl5k1*spak1l3
      abb67(21)=abb67(20)*spbl3k2
      abb67(22)=spbk2k1*spak1l3
      abb67(23)=abb67(22)*spbl5l3
      abb67(13)=abb67(13)-abb67(23)+abb67(19)-abb67(21)
      abb67(13)=abb67(13)*abb67(4)
      abb67(10)=abb67(10)-abb67(13)
      abb67(13)=c2-c1
      abb67(10)=abb67(13)*abb67(10)
      abb67(19)=abb67(1)**2
      abb67(21)=abb67(10)*abb67(19)
      abb67(24)=abb67(23)*abb67(6)
      abb67(24)=abb67(24)+abb67(12)
      abb67(24)=abb67(24)*spak2l4
      abb67(25)=spak2l3*spbl3k1
      abb67(26)=abb67(4)*spbl5k2
      abb67(27)=abb67(26)*spak1l5
      abb67(28)=abb67(25)*abb67(27)
      abb67(15)=-abb67(24)+abb67(28)+abb67(15)
      abb67(15)=abb67(15)*abb67(5)
      abb67(24)=abb67(12)*spbl5k2
      abb67(23)=abb67(24)+abb67(23)
      abb67(23)=abb67(23)*abb67(4)
      abb67(15)=abb67(15)-abb67(23)
      abb67(23)=mT*abb67(1)
      abb67(24)=abb67(13)*abb67(23)
      abb67(15)=-abb67(15)*abb67(24)
      abb67(15)=abb67(21)+abb67(15)
      abb67(15)=mT*abb67(15)
      abb67(21)=mH**2*abb67(7)*abb67(8)
      abb67(17)=abb67(17)*abb67(21)
      abb67(28)=spak2l3*spbl5l3
      abb67(29)=abb67(28)*spal4l5
      abb67(17)=abb67(17)-abb67(29)
      abb67(17)=abb67(17)*spbl5k2
      abb67(29)=spbl5k1*spal4l5
      abb67(30)=spak1l3*spbl5l3
      abb67(31)=abb67(29)*abb67(30)
      abb67(17)=abb67(17)+abb67(31)
      abb67(17)=-abb67(17)*abb67(13)
      abb67(31)=spak1l4*spbl5k1
      abb67(32)=spbl5k2*spak2l4
      abb67(31)=abb67(31)-abb67(32)
      abb67(31)=-abb67(31)*abb67(13)
      abb67(32)=abb67(31)*abb67(19)
      abb67(17)=abb67(32)+abb67(17)
      abb67(17)=abb67(1)*abb67(17)
      abb67(15)=abb67(17)+abb67(15)
      abb67(15)=abb67(3)*abb67(15)
      abb67(17)=abb67(26)*abb67(21)
      abb67(32)=abb67(5)*spak2l4
      abb67(33)=abb67(32)*abb67(21)
      abb67(34)=abb67(17)-abb67(33)
      abb67(34)=abb67(13)*abb67(34)
      abb67(35)=abb67(34)*abb67(19)
      abb67(36)=abb67(17)+abb67(33)
      abb67(36)=abb67(36)*abb67(13)
      abb67(37)=-abb67(5)*abb67(13)
      abb67(38)=-abb67(4)*abb67(37)
      abb67(39)=abb67(38)*abb67(19)
      abb67(36)=abb67(39)+abb67(36)
      abb67(36)=abb67(1)*abb67(36)
      abb67(39)=abb67(6)*spak2l4*abb67(5)**2
      abb67(40)=abb67(9)*spbl5k2*abb67(4)**2
      abb67(39)=abb67(39)-abb67(40)
      abb67(41)=abb67(13)*mT
      abb67(39)=abb67(41)*abb67(39)
      abb67(42)=abb67(19)*abb67(39)
      abb67(36)=abb67(36)+abb67(42)
      abb67(36)=mT*abb67(36)
      abb67(35)=abb67(35)+abb67(36)
      abb67(35)=mT*abb67(35)
      abb67(15)=abb67(35)+abb67(15)
      abb67(35)=e*gs**4*abb67(2)*gHT*spbe2e1*spae1e2*TR*i_
      abb67(36)=1.0_ki/2.0_ki*abb67(35)
      abb67(15)=abb67(15)*abb67(36)
      abb67(42)=abb67(38)*abb67(1)
      abb67(39)=abb67(39)+abb67(42)
      abb67(43)=mT**2
      abb67(44)=-abb67(43)*abb67(21)*abb67(39)
      abb67(12)=abb67(12)*abb67(4)
      abb67(12)=-abb67(12)+2.0_ki*spak2l4
      abb67(12)=abb67(12)*abb67(5)
      abb67(12)=abb67(12)+2.0_ki*abb67(26)
      abb67(12)=-abb67(12)*abb67(24)
      abb67(45)=abb67(32)-abb67(26)
      abb67(45)=abb67(13)*abb67(45)
      abb67(46)=3.0_ki*abb67(19)
      abb67(47)=-abb67(45)*abb67(46)
      abb67(12)=abb67(47)+abb67(12)
      abb67(12)=abb67(12)*abb67(3)*mT
      abb67(12)=abb67(44)+abb67(12)
      abb67(12)=abb67(12)*abb67(35)
      abb67(44)=abb67(35)*abb67(3)
      abb67(47)=abb67(44)*abb67(1)
      abb67(48)=abb67(43)*abb67(38)*abb67(47)
      abb67(49)=-4.0_ki*abb67(48)
      abb67(10)=mT*abb67(10)
      abb67(31)=abb67(1)*abb67(31)
      abb67(10)=abb67(31)+abb67(10)
      abb67(10)=abb67(3)*abb67(10)
      abb67(31)=mT*abb67(39)
      abb67(31)=abb67(31)+abb67(34)
      abb67(31)=mT*abb67(31)
      abb67(10)=abb67(31)+abb67(10)
      abb67(10)=abb67(10)*abb67(36)
      abb67(31)=abb67(44)*mT
      abb67(34)=-abb67(45)*abb67(31)
      abb67(39)=1.0_ki/2.0_ki*mT
      abb67(45)=abb67(39)*abb67(44)
      abb67(50)=abb67(13)*abb67(45)
      abb67(51)=spbl5k2*spak2l3
      abb67(20)=abb67(51)-abb67(20)
      abb67(26)=-abb67(20)*abb67(26)*abb67(50)
      abb67(17)=abb67(17)*abb67(16)*abb67(50)
      abb67(51)=abb67(4)*spbl5l4
      abb67(52)=abb67(51)*abb67(22)
      abb67(20)=abb67(52)-abb67(20)
      abb67(20)=abb67(20)*abb67(50)
      abb67(52)=abb67(13)*abb67(1)
      abb67(28)=abb67(28)*abb67(52)
      abb67(53)=mT*abb67(13)*abb67(21)
      abb67(16)=abb67(16)*abb67(53)
      abb67(16)=abb67(28)+abb67(16)
      abb67(16)=abb67(3)*abb67(16)
      abb67(28)=abb67(51)*abb67(53)
      abb67(16)=abb67(28)+abb67(16)
      abb67(16)=abb67(16)*abb67(36)
      abb67(28)=spak1k2*abb67(21)*spbl5k2
      abb67(28)=abb67(28)+abb67(30)
      abb67(28)=-abb67(28)*abb67(52)
      abb67(30)=abb67(37)*spak1k2
      abb67(36)=mT*abb67(30)*abb67(46)
      abb67(28)=abb67(28)+abb67(36)
      abb67(36)=1.0_ki/2.0_ki*abb67(44)
      abb67(28)=abb67(28)*abb67(36)
      abb67(30)=abb67(30)*abb67(45)
      abb67(51)=spak2l4*spbl3k2
      abb67(14)=abb67(51)-abb67(14)
      abb67(51)=-abb67(14)*abb67(52)
      abb67(53)=spal4l5*abb67(11)*abb67(5)
      abb67(53)=abb67(53)+abb67(14)
      abb67(53)=abb67(53)*abb67(41)
      abb67(51)=abb67(51)+abb67(53)
      abb67(51)=abb67(51)*abb67(36)
      abb67(14)=abb67(14)*abb67(32)*abb67(50)
      abb67(32)=abb67(32)*abb67(6)
      abb67(53)=abb67(32)+abb67(4)
      abb67(11)=mT**3*abb67(36)*abb67(37)*abb67(53)*abb67(11)
      abb67(53)=-abb67(39)*abb67(21)*spal4l5*abb67(37)
      abb67(21)=abb67(18)*abb67(21)
      abb67(39)=-abb67(39)*abb67(21)*abb67(13)
      abb67(54)=spbl5k2*spal4l5
      abb67(21)=abb67(54)+1.0_ki/2.0_ki*abb67(21)
      abb67(21)=abb67(21)*abb67(52)
      abb67(21)=abb67(21)+abb67(39)
      abb67(21)=abb67(3)*abb67(21)
      abb67(21)=abb67(53)+abb67(21)
      abb67(21)=abb67(21)*abb67(35)
      abb67(35)=abb67(13)*abb67(47)
      abb67(18)=-abb67(33)*abb67(18)*abb67(50)
      abb67(33)=abb67(5)*abb67(4)
      abb67(33)=abb67(40)+abb67(33)
      abb67(33)=abb67(33)*abb67(41)
      abb67(33)=-abb67(42)+abb67(33)
      abb67(22)=abb67(43)*abb67(36)*abb67(22)*abb67(33)
      abb67(27)=abb67(27)+spak1l4
      abb67(27)=-abb67(23)*abb67(27)*abb67(37)
      abb67(19)=3.0_ki/2.0_ki*abb67(19)
      abb67(33)=abb67(37)*spak1l4
      abb67(36)=-abb67(33)*abb67(19)
      abb67(27)=abb67(36)+abb67(27)
      abb67(27)=abb67(27)*abb67(31)
      abb67(31)=-abb67(33)*abb67(45)
      abb67(32)=abb67(32)*spbk2k1
      abb67(33)=abb67(4)*spbk2k1
      abb67(32)=abb67(32)+abb67(33)
      abb67(24)=abb67(32)*abb67(24)
      abb67(32)=abb67(13)*abb67(33)
      abb67(19)=abb67(32)*abb67(19)
      abb67(19)=abb67(19)+abb67(24)
      abb67(19)=mT*abb67(19)
      abb67(24)=-abb67(29)*abb67(52)
      abb67(19)=abb67(24)+abb67(19)
      abb67(19)=abb67(19)*abb67(44)
      abb67(24)=abb67(32)*abb67(45)
      abb67(23)=-abb67(23)*abb67(25)*abb67(38)
      abb67(13)=abb67(13)*abb67(4)*spbl5k1
      abb67(25)=-abb67(13)*abb67(46)
      abb67(23)=abb67(25)+abb67(23)
      abb67(23)=abb67(23)*abb67(45)
      abb67(13)=-abb67(13)*abb67(45)
      R2d67=0.0_ki
      rat2 = rat2 + R2d67
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='67' value='", &
          & R2d67, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd67h8
