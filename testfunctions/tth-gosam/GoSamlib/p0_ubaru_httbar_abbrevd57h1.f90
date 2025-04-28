module     p0_ubaru_httbar_abbrevd57h1
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh1
   implicit none
   private
   complex(ki), dimension(33), public :: abb57
   complex(ki), public :: R2d57
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_color, only: TR
      use p0_ubaru_httbar_globalsl1, only: epspow
      implicit none
      abb57(1)=sqrt(mT**2)
      abb57(2)=es12**(-1)
      abb57(3)=spbl4k2**(-1)
      abb57(4)=spbl5k2**(-1)
      abb57(5)=spak2l3**(-1)
      abb57(6)=spbl3k2**(-1)
      abb57(7)=spak2l5**(-1)
      abb57(8)=spak2l4**(-1)
      abb57(9)=spak1l3*spbk2k1
      abb57(10)=abb57(9)*abb57(4)
      abb57(10)=abb57(10)-spal3l5
      abb57(11)=spak1l4*mT
      abb57(10)=abb57(10)*abb57(11)
      abb57(9)=abb57(9)*abb57(3)
      abb57(9)=abb57(9)-spal3l4
      abb57(12)=spak1l5*mT
      abb57(9)=abb57(9)*abb57(12)
      abb57(13)=abb57(3)*spal3l5
      abb57(14)=spal3l4*abb57(4)
      abb57(13)=abb57(13)+abb57(14)
      abb57(14)=spbl3k2*spak1l3
      abb57(15)=abb57(14)*mT
      abb57(16)=abb57(13)*abb57(15)
      abb57(9)=abb57(16)+abb57(10)-abb57(9)
      abb57(9)=abb57(9)*spbl3k2
      abb57(10)=spak1l3*spal4l5
      abb57(16)=spak1l5*spal3l4
      abb57(17)=abb57(16)+2.0_ki*abb57(10)
      abb57(17)=abb57(17)*spbl3k2
      abb57(18)=spal4l5*spbl5k2
      abb57(19)=spak1l4*spbk2k1
      abb57(18)=-abb57(18)+2.0_ki*abb57(19)
      abb57(18)=abb57(18)*spak1l5
      abb57(20)=spak1l4*spbl4k2
      abb57(21)=abb57(20)*spal4l5
      abb57(18)=abb57(18)+abb57(21)
      abb57(17)=abb57(17)-abb57(18)
      abb57(17)=abb57(17)*abb57(1)
      abb57(9)=abb57(17)-abb57(9)
      abb57(17)=c2*NC
      abb57(17)=abb57(17)-c1
      abb57(21)=abb57(2)*gHT*e*i_*gs**4*TR**2
      abb57(22)=abb57(17)*abb57(21)
      abb57(9)=-abb57(1)*abb57(9)*abb57(22)
      abb57(23)=2.0_ki*abb57(21)
      abb57(24)=abb57(17)*abb57(23)
      abb57(25)=-abb57(16)*spbl3k2*abb57(24)
      abb57(26)=abb57(5)*mH**2
      abb57(27)=abb57(26)*abb57(6)
      abb57(19)=spak1l5*abb57(19)*abb57(27)
      abb57(28)=abb57(27)*abb57(20)
      abb57(29)=abb57(28)*spal4l5
      abb57(19)=abb57(19)-abb57(29)
      abb57(19)=abb57(19)*abb57(24)
      abb57(10)=abb57(10)+abb57(16)
      abb57(29)=2.0_ki*spbl3k2
      abb57(10)=abb57(10)*abb57(29)
      abb57(10)=abb57(10)-abb57(18)
      abb57(10)=-abb57(10)*abb57(22)
      abb57(14)=abb57(14)-abb57(20)
      abb57(18)=abb57(14)*spal3l4
      abb57(16)=abb57(16)*spbl5k2
      abb57(16)=abb57(16)+abb57(18)
      abb57(16)=abb57(16)*abb57(22)
      abb57(18)=abb57(15)*abb57(3)
      abb57(18)=abb57(18)-abb57(11)
      abb57(20)=abb57(1)*spak1l4
      abb57(18)=abb57(20)+2.0_ki*abb57(18)
      abb57(18)=abb57(18)*abb57(1)
      abb57(20)=abb57(28)*spak2l4
      abb57(29)=abb57(27)*spak2l4
      abb57(30)=spak1l5*spbl5k2
      abb57(31)=abb57(29)*abb57(30)
      abb57(32)=abb57(26)*spak2l4
      abb57(33)=abb57(32)*spak1l3
      abb57(18)=abb57(18)-abb57(31)-abb57(33)+abb57(20)
      abb57(18)=-abb57(18)*abb57(22)
      abb57(17)=-spak1l4*abb57(17)
      abb57(20)=abb57(23)*abb57(27)*abb57(17)
      abb57(17)=abb57(17)*abb57(21)
      abb57(14)=abb57(14)+abb57(30)
      abb57(14)=-abb57(14)*spal3l5*abb57(22)
      abb57(21)=spak1l3*abb57(26)
      abb57(21)=abb57(21)-abb57(28)
      abb57(21)=spak2l5*abb57(21)
      abb57(15)=abb57(15)*abb57(4)
      abb57(15)=abb57(15)+abb57(12)
      abb57(23)=abb57(1)*spak1l5
      abb57(15)=-abb57(23)+2.0_ki*abb57(15)
      abb57(15)=abb57(15)*abb57(1)
      abb57(23)=abb57(27)*spak2l5
      abb57(26)=abb57(23)*abb57(30)
      abb57(15)=abb57(15)+abb57(26)+abb57(21)
      abb57(15)=-abb57(15)*abb57(22)
      abb57(21)=spak1l5*abb57(22)
      abb57(26)=abb57(22)*spal4l5
      abb57(27)=spbl3k2*spak1l5
      abb57(28)=abb57(27)*abb57(26)
      abb57(12)=abb57(12)*abb57(3)
      abb57(11)=abb57(11)*abb57(4)
      abb57(11)=abb57(12)-abb57(11)
      abb57(12)=abb57(11)*abb57(1)
      abb57(30)=mT**2
      abb57(31)=abb57(8)*abb57(30)*abb57(3)
      abb57(31)=abb57(31)+1.0_ki
      abb57(31)=spak1l4*abb57(31)*spak2l5
      abb57(30)=abb57(7)*abb57(30)*abb57(4)
      abb57(30)=abb57(30)+1.0_ki
      abb57(30)=spak1l5*abb57(30)*spak2l4
      abb57(12)=abb57(12)+abb57(31)-abb57(30)
      abb57(12)=abb57(12)*spbl3k2
      abb57(30)=spak1l4*spbl3k1
      abb57(30)=abb57(30)-abb57(32)
      abb57(30)=abb57(30)*spak1l5
      abb57(31)=spbl4l3*spak1l4*spal4l5
      abb57(12)=abb57(12)+abb57(30)-abb57(31)
      abb57(12)=abb57(12)*abb57(22)
      abb57(22)=abb57(27)*abb57(17)
      abb57(11)=-spbk2k1*abb57(11)
      abb57(13)=abb57(13)*spbl3k2*mT
      abb57(27)=2.0_ki*abb57(1)
      abb57(30)=abb57(27)*spal4l5
      abb57(11)=abb57(11)-abb57(30)+abb57(13)
      abb57(11)=abb57(1)*abb57(11)*abb57(24)
      abb57(13)=-4.0_ki*abb57(26)
      abb57(26)=spal3l4*abb57(24)
      abb57(27)=abb57(27)*mT
      abb57(30)=abb57(27)*abb57(3)
      abb57(29)=abb57(30)-abb57(29)
      abb57(29)=-abb57(29)*abb57(24)
      abb57(30)=-spal3l5*abb57(24)
      abb57(27)=abb57(27)*abb57(4)
      abb57(23)=abb57(27)+abb57(23)
      abb57(23)=-abb57(23)*abb57(24)
      R2d57=0.0_ki
      rat2 = rat2 + R2d57
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='57' value='", &
          & R2d57, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd57h1
