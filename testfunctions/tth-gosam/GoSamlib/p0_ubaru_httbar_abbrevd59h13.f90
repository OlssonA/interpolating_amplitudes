module     p0_ubaru_httbar_abbrevd59h13
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh13
   implicit none
   private
   complex(ki), dimension(34), public :: abb59
   complex(ki), public :: R2d59
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
      abb59(1)=sqrt(mT**2)
      abb59(2)=NC**(-1)
      abb59(3)=es12**(-1)
      abb59(4)=spak2l4**(-1)
      abb59(5)=spak2l5**(-1)
      abb59(6)=spak2l3**(-1)
      abb59(7)=spbl3k2**(-1)
      abb59(8)=spbl4k2**(-1)
      abb59(9)=spbl3k2*spak2l3
      abb59(10)=abb59(9)*abb59(4)
      abb59(11)=spak1k2*abb59(5)
      abb59(12)=abb59(10)*abb59(11)
      abb59(13)=spbl5k2*spak1k2
      abb59(14)=abb59(4)**2*abb59(13)*abb59(8)
      abb59(9)=abb59(14)*abb59(9)
      abb59(9)=abb59(12)+abb59(9)
      abb59(12)=mT**2
      abb59(9)=abb59(9)*abb59(12)
      abb59(15)=spbl4k2*abb59(5)
      abb59(16)=abb59(7)*abb59(6)*spak1k2*mH**2
      abb59(17)=abb59(15)*abb59(16)
      abb59(18)=spak1l3*spbl4l3
      abb59(19)=abb59(18)*abb59(5)
      abb59(17)=abb59(19)+abb59(17)
      abb59(19)=spbl4k2*abb59(17)*spak2l4
      abb59(20)=spbl5l4*spak1l4
      abb59(21)=abb59(10)*abb59(20)
      abb59(22)=abb59(16)*spbl4k2
      abb59(23)=abb59(22)+abb59(18)
      abb59(24)=abb59(23)*spbl5k2
      abb59(9)=abb59(9)+abb59(19)+abb59(21)+abb59(24)
      abb59(9)=abb59(9)*mT
      abb59(19)=c2*abb59(9)
      abb59(21)=abb59(15)*spak1k2
      abb59(13)=abb59(13)*abb59(4)
      abb59(13)=abb59(21)+abb59(13)
      abb59(21)=abb59(1)*mT
      abb59(13)=abb59(13)*abb59(21)
      abb59(25)=spbl5l4*spak1l5
      abb59(18)=abb59(25)+abb59(18)
      abb59(18)=abb59(18)*spbl5k2
      abb59(25)=spak1l3*spbl5l3
      abb59(26)=abb59(20)-abb59(25)
      abb59(26)=abb59(26)*spbl4k2
      abb59(13)=abb59(13)+abb59(18)+abb59(26)
      abb59(18)=abb59(13)*c2
      abb59(26)=abb59(1)*abb59(18)
      abb59(19)=abb59(19)+abb59(26)
      abb59(26)=abb59(3)*gHT*e*i_*gs**4*TR**2
      abb59(27)=abb59(26)*abb59(1)
      abb59(19)=abb59(19)*abb59(27)
      abb59(9)=-c1*abb59(9)
      abb59(13)=c1*abb59(13)
      abb59(28)=-abb59(1)*abb59(13)
      abb59(9)=abb59(9)+abb59(28)
      abb59(28)=abb59(26)*abb59(2)
      abb59(29)=abb59(28)*abb59(1)
      abb59(9)=abb59(9)*abb59(29)
      abb59(9)=abb59(19)+abb59(9)
      abb59(19)=2.0_ki*abb59(2)
      abb59(9)=abb59(9)*abb59(19)
      abb59(30)=abb59(28)*c1
      abb59(31)=abb59(26)*c2
      abb59(31)=abb59(30)-abb59(31)
      abb59(32)=4.0_ki*abb59(2)
      abb59(33)=abb59(31)*abb59(32)
      abb59(24)=-abb59(24)*abb59(33)
      abb59(22)=abb59(22)*spbl5k2
      abb59(34)=abb59(25)*spbl4k2
      abb59(22)=abb59(22)+abb59(34)
      abb59(22)=abb59(22)*abb59(33)
      abb59(33)=abb59(11)*abb59(4)
      abb59(14)=abb59(14)+abb59(33)
      abb59(12)=2.0_ki*abb59(12)
      abb59(12)=abb59(14)*abb59(12)
      abb59(14)=spbl5k2*abb59(4)
      abb59(33)=abb59(14)*abb59(16)
      abb59(20)=abb59(20)*abb59(4)
      abb59(12)=abb59(12)-abb59(17)+abb59(33)+2.0_ki*abb59(20)
      abb59(17)=abb59(30)*abb59(21)
      abb59(20)=c2*abb59(21)*abb59(26)
      abb59(17)=abb59(17)-abb59(20)
      abb59(12)=abb59(32)*abb59(12)*abb59(17)
      abb59(18)=abb59(18)*abb59(26)
      abb59(13)=-abb59(13)*abb59(28)
      abb59(13)=abb59(18)+abb59(13)
      abb59(13)=abb59(13)*abb59(19)
      abb59(18)=-abb59(19)*spbl5k2*abb59(31)
      abb59(16)=abb59(16)*spbl5k2
      abb59(16)=abb59(16)+abb59(25)
      abb59(16)=-abb59(16)*abb59(18)
      abb59(20)=abb59(19)*abb59(31)*spbl4k2
      abb59(21)=-abb59(23)*abb59(20)
      abb59(23)=-abb59(17)*abb59(19)
      abb59(11)=-abb59(11)*spbl3k2*abb59(23)
      abb59(14)=-abb59(14)*spak1l3*abb59(23)
      abb59(10)=abb59(10)*mT
      abb59(23)=-c2*abb59(10)
      abb59(25)=3.0_ki*abb59(1)
      abb59(26)=abb59(25)*c2
      abb59(28)=-spbl4k2*abb59(26)
      abb59(23)=abb59(23)+abb59(28)
      abb59(23)=abb59(23)*abb59(27)
      abb59(10)=c1*abb59(10)
      abb59(25)=abb59(25)*c1
      abb59(28)=spbl4k2*abb59(25)
      abb59(10)=abb59(10)+abb59(28)
      abb59(10)=abb59(10)*abb59(29)
      abb59(10)=abb59(23)+abb59(10)
      abb59(10)=abb59(10)*abb59(19)
      abb59(23)=8.0_ki*abb59(2)
      abb59(17)=-abb59(23)*abb59(17)
      abb59(23)=abb59(4)*abb59(17)
      abb59(15)=abb59(15)*spak2l4
      abb59(15)=abb59(15)+spbl5k2
      abb59(28)=2.0_ki*mT
      abb59(15)=abb59(15)*abb59(28)
      abb59(28)=c2*abb59(15)
      abb59(26)=spbl5k2*abb59(26)
      abb59(26)=abb59(28)+abb59(26)
      abb59(26)=abb59(26)*abb59(27)
      abb59(15)=-c1*abb59(15)
      abb59(25)=-spbl5k2*abb59(25)
      abb59(15)=abb59(15)+abb59(25)
      abb59(15)=abb59(15)*abb59(29)
      abb59(15)=abb59(26)+abb59(15)
      abb59(15)=abb59(15)*abb59(19)
      abb59(17)=abb59(5)*abb59(17)
      R2d59=0.0_ki
      rat2 = rat2 + R2d59
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='59' value='", &
          & R2d59, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd59h13
