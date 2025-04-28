module     p2_gg_httbar_abbrevd50h12
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh12
   implicit none
   private
   complex(ki), dimension(38), public :: abb50
   complex(ki), public :: R2d50
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
      abb50(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb50(2)=es12**(-1)
      abb50(3)=spak2l4**(-1)
      abb50(4)=spak2l3**(-1)
      abb50(5)=spbl3k2**(-1)
      abb50(6)=spak2l5**(-1)
      abb50(7)=sqrt(mT**2)
      abb50(8)=1.0_ki/(-mT**2+es34)
      abb50(9)=c2-c1
      abb50(10)=mH**2*abb50(4)*abb50(5)
      abb50(11)=-abb50(9)*abb50(10)*spbl5k2
      abb50(12)=-abb50(1)*spbl4k1*abb50(11)
      abb50(10)=abb50(9)*abb50(10)*spbl4k2
      abb50(13)=-abb50(8)*spbl5k1*abb50(10)
      abb50(12)=abb50(12)-abb50(13)
      abb50(13)=-abb50(12)*spak1k2
      abb50(14)=abb50(6)*spbl4k1
      abb50(15)=abb50(14)*abb50(7)
      abb50(16)=abb50(3)*abb50(7)
      abb50(17)=abb50(16)*spbl5k1
      abb50(15)=abb50(15)+abb50(17)
      abb50(15)=-spak1k2*abb50(15)
      abb50(17)=abb50(6)*spak2l3
      abb50(18)=abb50(17)*spbl3k1
      abb50(18)=abb50(18)+spbl5k1
      abb50(18)=abb50(18)*abb50(3)
      abb50(18)=abb50(18)+abb50(14)
      abb50(19)=mT*spak1k2
      abb50(20)=-abb50(19)*abb50(18)
      abb50(15)=abb50(15)+abb50(20)
      abb50(20)=abb50(1)+abb50(8)
      abb50(21)=abb50(9)*abb50(20)
      abb50(15)=mT*abb50(21)*abb50(15)
      abb50(13)=abb50(13)+abb50(15)
      abb50(15)=spak2l3*spbl5l3
      abb50(22)=abb50(15)*spbl4k2
      abb50(23)=spbl4k1*spbl5l3
      abb50(24)=abb50(23)*spak1l3
      abb50(22)=abb50(22)-abb50(24)
      abb50(22)=-abb50(1)*abb50(22)*abb50(9)
      abb50(24)=spak2l3*spbl4l3
      abb50(25)=abb50(24)*spbl5k2
      abb50(26)=spbl5k1*spbl4l3
      abb50(27)=abb50(26)*spak1l3
      abb50(25)=abb50(25)-abb50(27)
      abb50(25)=-abb50(8)*abb50(25)*abb50(9)
      abb50(22)=abb50(22)+abb50(25)
      abb50(25)=-abb50(13)+abb50(22)
      abb50(27)=i_*spbe2e1*gs**4*gHT*e*spae1e2*TR
      abb50(28)=abb50(27)*abb50(2)
      abb50(29)=mT**2
      abb50(30)=abb50(29)*abb50(28)
      abb50(30)=-1.0_ki/6.0_ki*abb50(27)+abb50(30)
      abb50(25)=abb50(2)*abb50(25)*abb50(30)
      abb50(30)=abb50(7)**2
      abb50(22)=abb50(30)*abb50(22)
      abb50(12)=spak1k2*abb50(30)*abb50(12)
      abb50(18)=abb50(19)*abb50(30)*abb50(18)
      abb50(30)=spbl5k1*abb50(3)
      abb50(14)=abb50(30)+abb50(14)
      abb50(14)=spak1k2*abb50(14)*abb50(7)**3
      abb50(14)=abb50(14)+abb50(18)
      abb50(14)=mT*abb50(21)*abb50(14)
      abb50(12)=abb50(14)+abb50(12)+abb50(22)
      abb50(14)=abb50(27)*abb50(2)**2
      abb50(12)=abb50(12)*abb50(14)
      abb50(13)=-abb50(13)*abb50(14)
      abb50(18)=abb50(26)*abb50(9)*abb50(8)
      abb50(22)=abb50(1)*abb50(23)*abb50(9)
      abb50(18)=abb50(18)+abb50(22)
      abb50(22)=1.0_ki/2.0_ki*abb50(14)
      abb50(23)=abb50(22)*spak1k2
      abb50(18)=abb50(18)*abb50(23)
      abb50(26)=-abb50(8)*spbl4l3*abb50(9)
      abb50(30)=-spbl5k2*abb50(26)
      abb50(31)=-abb50(1)*spbl5l3*abb50(9)
      abb50(32)=-spbl4k2*abb50(31)
      abb50(30)=abb50(30)+abb50(32)
      abb50(23)=abb50(30)*abb50(23)
      abb50(30)=-mT*abb50(3)
      abb50(30)=abb50(30)-abb50(16)
      abb50(30)=mT*abb50(21)*abb50(30)
      abb50(10)=-abb50(8)*abb50(10)
      abb50(10)=abb50(30)+abb50(10)
      abb50(30)=-abb50(10)*abb50(27)
      abb50(32)=abb50(28)*spak1l3*spbk2k1
      abb50(33)=-abb50(26)*abb50(32)
      abb50(30)=abb50(30)+abb50(33)
      abb50(33)=1.0_ki/2.0_ki*abb50(2)
      abb50(30)=abb50(30)*abb50(33)
      abb50(34)=2.0_ki*abb50(14)
      abb50(35)=abb50(10)*abb50(34)
      abb50(26)=abb50(14)*abb50(26)
      abb50(36)=-mT*abb50(6)
      abb50(37)=abb50(6)*abb50(7)
      abb50(36)=abb50(36)-abb50(37)
      abb50(36)=mT*abb50(21)*abb50(36)
      abb50(38)=abb50(11)*abb50(1)
      abb50(36)=abb50(36)+abb50(38)
      abb50(27)=-abb50(36)*abb50(27)
      abb50(32)=-abb50(31)*abb50(32)
      abb50(27)=abb50(27)+abb50(32)
      abb50(27)=abb50(27)*abb50(33)
      abb50(32)=abb50(36)*abb50(34)
      abb50(31)=abb50(14)*abb50(31)
      abb50(29)=-abb50(29)*abb50(17)*abb50(3)*abb50(21)
      abb50(28)=-1.0_ki/2.0_ki*abb50(28)*abb50(29)
      abb50(33)=abb50(34)*abb50(29)
      abb50(16)=abb50(16)*spbl5k2
      abb50(34)=abb50(37)*spbl4k2
      abb50(16)=abb50(16)+abb50(34)
      abb50(16)=spak1k2*abb50(16)
      abb50(17)=abb50(17)*spbl3k2
      abb50(17)=abb50(17)+spbl5k2
      abb50(17)=abb50(17)*abb50(3)
      abb50(34)=abb50(6)*spbl4k2
      abb50(17)=abb50(17)+abb50(34)
      abb50(17)=abb50(19)*abb50(17)
      abb50(16)=abb50(16)+abb50(17)
      abb50(16)=mT*abb50(21)*abb50(16)
      abb50(11)=-abb50(11)*spbl4k2*abb50(20)*spak1k2
      abb50(11)=abb50(11)+abb50(16)
      abb50(11)=abb50(11)*abb50(22)
      abb50(9)=-abb50(22)*spbk2k1*abb50(9)
      abb50(16)=-abb50(8)*abb50(24)*abb50(9)
      abb50(10)=-abb50(10)*abb50(14)
      abb50(9)=-abb50(1)*abb50(15)*abb50(9)
      abb50(15)=-abb50(36)*abb50(14)
      abb50(14)=-abb50(14)*abb50(29)
      R2d50=abb50(25)
      rat2 = rat2 + R2d50
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='50' value='", &
          & R2d50, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd50h12
