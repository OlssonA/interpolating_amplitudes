module     p0_ubaru_httbar_abbrevd57h14
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh14
   implicit none
   private
   complex(ki), dimension(39), public :: abb57
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
      abb57(3)=spak2l4**(-1)
      abb57(4)=spak2l5**(-1)
      abb57(5)=spak2l3**(-1)
      abb57(6)=spbl3k2**(-1)
      abb57(7)=spbl5k2**(-1)
      abb57(8)=spbl4k2**(-1)
      abb57(9)=NC*c2
      abb57(9)=c1-abb57(9)
      abb57(9)=abb57(9)*i_*e*gHT*abb57(2)*TR**2*gs**4
      abb57(10)=2.0_ki*abb57(9)
      abb57(11)=spbl4k1*spbl5k1
      abb57(12)=abb57(10)*abb57(11)
      abb57(13)=abb57(1)**2
      abb57(14)=abb57(13)*abb57(12)
      abb57(15)=spbl5k1*spak2l3
      abb57(16)=abb57(9)*abb57(15)
      abb57(17)=mT*abb57(1)
      abb57(18)=abb57(17)*abb57(3)
      abb57(19)=-spbl3k1*abb57(18)*abb57(16)
      abb57(20)=abb57(9)*spbl4k1
      abb57(21)=abb57(17)*abb57(4)
      abb57(22)=abb57(20)*abb57(21)
      abb57(23)=spbl3k1*spak2l3
      abb57(24)=abb57(23)*abb57(22)
      abb57(14)=abb57(24)+abb57(14)+abb57(19)
      abb57(14)=spak1k2*abb57(14)
      abb57(19)=spak2l5*spbl5k1
      abb57(24)=spak2l4*spbl4k1
      abb57(25)=abb57(19)-abb57(24)
      abb57(26)=2.0_ki*abb57(23)
      abb57(27)=-abb57(26)-abb57(25)
      abb57(28)=abb57(13)*spbl5l4
      abb57(27)=abb57(28)*abb57(27)
      abb57(29)=spbl3k1*spak2l3**2
      abb57(30)=spbl4l3*abb57(21)*abb57(29)
      abb57(27)=abb57(30)+abb57(27)
      abb57(27)=abb57(9)*abb57(27)
      abb57(30)=abb57(16)*spbl4l3
      abb57(31)=-abb57(13)+abb57(17)
      abb57(31)=abb57(31)*abb57(30)
      abb57(32)=-abb57(20)*abb57(17)*spak2l3
      abb57(33)=abb57(9)*abb57(18)
      abb57(29)=abb57(33)*abb57(29)
      abb57(29)=abb57(32)+abb57(29)
      abb57(29)=spbl5l3*abb57(29)
      abb57(14)=abb57(14)+abb57(29)+abb57(31)+abb57(27)
      abb57(27)=abb57(10)*spbl4l3
      abb57(29)=-abb57(15)*abb57(27)
      abb57(31)=abb57(6)*mH**2
      abb57(32)=abb57(31)*abb57(5)
      abb57(34)=abb57(32)*abb57(9)
      abb57(35)=abb57(34)*abb57(24)
      abb57(36)=-spbl5l4*abb57(35)
      abb57(37)=abb57(34)*abb57(11)*spak1k2
      abb57(36)=abb57(36)+abb57(37)
      abb57(36)=2.0_ki*abb57(36)
      abb57(37)=abb57(9)*spbl5l4
      abb57(38)=-abb57(23)*abb57(37)
      abb57(30)=abb57(38)-abb57(30)
      abb57(38)=-abb57(37)*abb57(25)
      abb57(12)=spak1k2*abb57(12)
      abb57(12)=abb57(12)+2.0_ki*abb57(30)+abb57(38)
      abb57(30)=spbl5l4*abb57(16)
      abb57(25)=abb57(23)+abb57(25)
      abb57(38)=abb57(9)*spbl4l3
      abb57(25)=abb57(38)*abb57(25)
      abb57(23)=abb57(24)-abb57(19)-abb57(23)
      abb57(23)=abb57(23)*abb57(9)*spbl5l3
      abb57(19)=abb57(19)*abb57(32)
      abb57(24)=-abb57(24)*abb57(32)
      abb57(39)=abb57(31)*spbl3k1
      abb57(24)=abb57(39)+abb57(19)+abb57(24)
      abb57(24)=spbl4k2*abb57(24)
      abb57(13)=-abb57(13)+2.0_ki*abb57(17)
      abb57(17)=spbl4k1*abb57(13)
      abb57(39)=-abb57(18)*abb57(26)
      abb57(17)=abb57(39)+abb57(17)+abb57(24)
      abb57(17)=abb57(9)*abb57(17)
      abb57(24)=-spbl4k1*abb57(32)*abb57(10)
      abb57(13)=-spbl5k1*abb57(13)
      abb57(26)=-abb57(21)*abb57(26)
      abb57(19)=-spbl5k2*abb57(19)
      abb57(13)=abb57(19)+abb57(26)+abb57(13)
      abb57(13)=abb57(9)*abb57(13)
      abb57(19)=abb57(9)*abb57(31)
      abb57(26)=-spbl3k1*abb57(19)
      abb57(26)=abb57(26)+abb57(35)
      abb57(26)=spbl5k2*abb57(26)
      abb57(13)=abb57(26)+abb57(13)
      abb57(26)=abb57(9)*spbl5k1
      abb57(31)=mT**2
      abb57(32)=abb57(8)*abb57(3)*abb57(31)
      abb57(32)=abb57(32)+1.0_ki
      abb57(32)=abb57(32)*spbl5k2*spak2l3
      abb57(35)=abb57(21)*spak2l3
      abb57(39)=-spal3l4*spbl5l4
      abb57(32)=abb57(32)+abb57(39)-abb57(35)
      abb57(32)=spbl4k1*abb57(32)
      abb57(31)=-abb57(7)*spbl4k2*abb57(4)*abb57(31)
      abb57(31)=abb57(31)+abb57(18)
      abb57(15)=abb57(15)*abb57(31)
      abb57(11)=spak1l3*abb57(11)
      abb57(11)=abb57(11)+abb57(15)+abb57(32)
      abb57(9)=abb57(9)*abb57(11)
      abb57(11)=-spbl5k1*abb57(19)
      abb57(11)=-abb57(16)+abb57(11)
      abb57(11)=spbl4k2*abb57(11)
      abb57(9)=abb57(11)+abb57(9)
      abb57(11)=-spbl5k1*abb57(33)
      abb57(11)=abb57(11)+abb57(22)
      abb57(11)=spak1k2*abb57(11)
      abb57(15)=-abb57(28)*abb57(10)
      abb57(19)=abb57(35)*abb57(38)
      abb57(22)=spbl5l3*spak2l3*abb57(33)
      abb57(11)=abb57(11)+abb57(22)+abb57(15)+abb57(19)
      abb57(11)=2.0_ki*abb57(11)
      abb57(15)=-4.0_ki*abb57(37)
      abb57(19)=-spbl5l3*abb57(10)
      abb57(18)=-abb57(18)*abb57(10)
      abb57(22)=spbl4k2*abb57(34)
      abb57(18)=abb57(18)+abb57(22)
      abb57(18)=2.0_ki*abb57(18)
      abb57(10)=-abb57(21)*abb57(10)
      abb57(21)=-spbl5k2*abb57(34)
      abb57(10)=abb57(10)+abb57(21)
      abb57(10)=2.0_ki*abb57(10)
      abb57(16)=-spbl4k1*abb57(16)
      R2d57=0.0_ki
      rat2 = rat2 + R2d57
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='57' value='", &
          & R2d57, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd57h14
