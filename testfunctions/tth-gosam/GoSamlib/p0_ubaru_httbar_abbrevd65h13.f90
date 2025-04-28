module     p0_ubaru_httbar_abbrevd65h13
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh13
   implicit none
   private
   complex(ki), dimension(32), public :: abb65
   complex(ki), public :: R2d65
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
      abb65(1)=1.0_ki/(-mT**2+es34)
      abb65(2)=NC**(-1)
      abb65(3)=spak2l3**(-1)
      abb65(4)=spbl3k2**(-1)
      abb65(5)=spak2l4**(-1)
      abb65(6)=spak2l5**(-1)
      abb65(7)=sqrt(mT**2)
      abb65(8)=mH**2*spak1k2*abb65(3)*abb65(4)*spbl4k2
      abb65(9)=spbl4l3*spak1l3
      abb65(8)=abb65(8)+abb65(9)
      abb65(9)=2.0_ki*c2
      abb65(10)=abb65(8)*abb65(9)
      abb65(11)=c1*abb65(8)
      abb65(12)=abb65(11)*abb65(2)
      abb65(10)=abb65(10)-abb65(12)
      abb65(10)=abb65(10)*abb65(2)
      abb65(13)=NC*abb65(8)
      abb65(14)=abb65(13)*c2
      abb65(10)=abb65(10)-abb65(14)
      abb65(10)=abb65(10)*spbl5k2
      abb65(14)=abb65(2)*c1
      abb65(15)=abb65(5)*spak1k2
      abb65(16)=abb65(14)*abb65(15)
      abb65(17)=c2*abb65(5)
      abb65(18)=abb65(17)*spak1k2
      abb65(19)=2.0_ki*abb65(18)
      abb65(16)=abb65(16)-abb65(19)
      abb65(16)=abb65(16)*abb65(2)
      abb65(18)=abb65(18)*NC
      abb65(16)=abb65(16)+abb65(18)
      abb65(20)=abb65(16)*spbl5k2
      abb65(21)=mT**2
      abb65(22)=abb65(21)*abb65(20)
      abb65(22)=-abb65(10)+abb65(22)
      abb65(23)=abb65(1)*gs**4*TR**2*gHT*e
      abb65(24)=abb65(23)*i_
      abb65(24)=4.0_ki*abb65(24)
      abb65(22)=abb65(22)*abb65(24)
      abb65(25)=abb65(8)*abb65(6)
      abb65(26)=abb65(7)**2
      abb65(27)=abb65(6)*abb65(26)*abb65(15)
      abb65(25)=-abb65(25)+abb65(27)
      abb65(25)=abb65(25)*abb65(9)
      abb65(11)=abb65(6)*abb65(11)
      abb65(28)=-c1*abb65(27)
      abb65(11)=abb65(11)+abb65(28)
      abb65(11)=abb65(2)*abb65(11)
      abb65(11)=abb65(25)+abb65(11)
      abb65(11)=abb65(2)*abb65(11)
      abb65(13)=abb65(13)*abb65(6)
      abb65(25)=-NC*abb65(27)
      abb65(25)=abb65(13)+abb65(25)
      abb65(25)=c2*abb65(25)
      abb65(27)=abb65(14)*abb65(6)
      abb65(15)=abb65(15)*abb65(27)
      abb65(19)=-abb65(6)*abb65(19)
      abb65(15)=abb65(19)+abb65(15)
      abb65(15)=abb65(2)*abb65(15)
      abb65(18)=abb65(6)*abb65(18)
      abb65(15)=abb65(18)+abb65(15)
      abb65(15)=abb65(15)*abb65(21)
      abb65(11)=abb65(15)-abb65(20)+abb65(25)+abb65(11)
      abb65(11)=mT*abb65(11)
      abb65(15)=abb65(9)*abb65(7)
      abb65(18)=abb65(15)*abb65(6)
      abb65(8)=abb65(8)*abb65(18)
      abb65(12)=-abb65(6)*abb65(7)*abb65(12)
      abb65(8)=abb65(8)+abb65(12)
      abb65(8)=abb65(2)*abb65(8)
      abb65(12)=-spbl5k2*abb65(7)*abb65(16)
      abb65(19)=c2*abb65(7)
      abb65(13)=-abb65(19)*abb65(13)
      abb65(8)=abb65(11)+abb65(12)+abb65(13)+abb65(8)
      abb65(8)=mT*abb65(8)
      abb65(8)=abb65(10)+abb65(8)
      abb65(8)=abb65(8)*abb65(24)
      abb65(11)=mT*spbl5k2
      abb65(13)=abb65(16)*abb65(11)
      abb65(12)=-abb65(12)+abb65(13)
      abb65(12)=mT*abb65(12)
      abb65(10)=-abb65(10)+abb65(12)
      abb65(10)=abb65(10)*abb65(24)
      abb65(12)=abb65(14)*abb65(7)
      abb65(13)=abb65(5)*abb65(12)
      abb65(16)=abb65(17)*abb65(7)
      abb65(13)=-abb65(13)+2.0_ki*abb65(16)
      abb65(13)=abb65(13)*abb65(2)
      abb65(16)=abb65(16)*NC
      abb65(13)=abb65(13)-abb65(16)
      abb65(16)=2.0_ki*i_
      abb65(16)=abb65(16)*abb65(23)
      abb65(20)=abb65(11)*abb65(16)
      abb65(23)=-abb65(20)*spak1l5*abb65(13)
      abb65(19)=abb65(19)*NC
      abb65(24)=abb65(6)*spak1l5
      abb65(25)=abb65(24)*abb65(19)
      abb65(12)=abb65(15)-abb65(12)
      abb65(15)=abb65(24)*abb65(2)
      abb65(28)=-abb65(12)*abb65(15)
      abb65(25)=abb65(25)+abb65(28)
      abb65(25)=spbl5k2*abb65(25)
      abb65(28)=c2*NC
      abb65(29)=abb65(28)*abb65(24)
      abb65(30)=abb65(14)-abb65(9)
      abb65(31)=abb65(30)*abb65(15)
      abb65(29)=abb65(29)+abb65(31)
      abb65(29)=abb65(29)*abb65(11)
      abb65(25)=abb65(25)+abb65(29)
      abb65(29)=abb65(16)*mT
      abb65(25)=abb65(25)*abb65(29)
      abb65(17)=abb65(17)*spak2l3
      abb65(31)=abb65(17)*NC
      abb65(24)=abb65(24)*abb65(31)
      abb65(32)=abb65(5)*spak2l3
      abb65(14)=abb65(14)*abb65(32)
      abb65(17)=2.0_ki*abb65(17)
      abb65(14)=-abb65(17)+abb65(14)
      abb65(14)=abb65(14)*abb65(15)
      abb65(14)=abb65(24)+abb65(14)
      abb65(15)=abb65(21)*abb65(16)
      abb65(14)=spbl5k2*abb65(14)*abb65(15)
      abb65(21)=abb65(20)*spak2l5*abb65(13)
      abb65(24)=abb65(30)*abb65(2)
      abb65(24)=-abb65(28)-abb65(24)
      abb65(24)=spbl5k2*abb65(26)*abb65(24)
      abb65(12)=abb65(12)*abb65(2)
      abb65(12)=-abb65(19)+abb65(12)
      abb65(11)=abb65(12)*abb65(11)
      abb65(11)=abb65(24)+abb65(11)
      abb65(11)=abb65(11)*abb65(16)
      abb65(12)=abb65(20)*spak2l3*abb65(13)
      abb65(13)=abb65(7)*abb65(27)
      abb65(13)=-abb65(18)+abb65(13)
      abb65(13)=abb65(2)*abb65(13)
      abb65(9)=-abb65(6)*abb65(9)
      abb65(9)=abb65(9)+abb65(27)
      abb65(9)=abb65(2)*abb65(9)
      abb65(16)=abb65(6)*abb65(28)
      abb65(9)=abb65(16)+abb65(9)
      abb65(9)=mT*abb65(9)
      abb65(16)=abb65(6)*abb65(19)
      abb65(9)=abb65(9)+abb65(16)+abb65(13)
      abb65(9)=abb65(9)*abb65(29)
      abb65(13)=abb65(27)*abb65(32)
      abb65(16)=-abb65(6)*abb65(17)
      abb65(13)=abb65(16)+abb65(13)
      abb65(13)=abb65(2)*abb65(13)
      abb65(16)=abb65(6)*abb65(31)
      abb65(13)=abb65(16)+abb65(13)
      abb65(13)=abb65(13)*abb65(15)
      R2d65=0.0_ki
      rat2 = rat2 + R2d65
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='65' value='", &
          & R2d65, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd65h13
