module     p0_ubaru_httbar_abbrevd59h5
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh5
   implicit none
   private
   complex(ki), dimension(42), public :: abb59
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
      abb59(5)=spbl5k2**(-1)
      abb59(6)=spak2l3**(-1)
      abb59(7)=spbl3k2**(-1)
      abb59(8)=spbl4k2**(-1)
      abb59(9)=spak2l5**(-1)
      abb59(10)=i_*e*gHT*abb59(3)*TR**2*gs**4
      abb59(11)=abb59(10)*abb59(4)
      abb59(12)=abb59(2)**2
      abb59(13)=abb59(11)*abb59(12)
      abb59(14)=c1*abb59(13)
      abb59(15)=abb59(14)*mT
      abb59(16)=abb59(11)*c2
      abb59(17)=mT*abb59(2)
      abb59(18)=abb59(17)*abb59(16)
      abb59(15)=abb59(15)-abb59(18)
      abb59(18)=abb59(1)**2
      abb59(19)=-abb59(18)*abb59(15)
      abb59(20)=spal3l5*spbl3k2
      abb59(21)=spal4l5*spbl4k2
      abb59(22)=abb59(21)-abb59(20)
      abb59(23)=-abb59(19)*abb59(22)
      abb59(24)=c1*abb59(12)*abb59(10)
      abb59(25)=abb59(10)*c2
      abb59(26)=abb59(25)*abb59(2)
      abb59(26)=abb59(26)-abb59(24)
      abb59(27)=abb59(1)*abb59(26)
      abb59(28)=abb59(6)*abb59(7)*mH**2
      abb59(29)=spbl4k2**2
      abb59(30)=-spal4l5*abb59(27)*abb59(29)*abb59(28)
      abb59(23)=abb59(30)+abb59(23)
      abb59(23)=spak1k2*abb59(23)
      abb59(30)=abb59(11)*abb59(1)
      abb59(31)=abb59(30)*abb59(17)
      abb59(32)=mT**2
      abb59(33)=abb59(32)*abb59(2)
      abb59(11)=abb59(33)*abb59(11)
      abb59(34)=abb59(11)+abb59(31)
      abb59(34)=c2*abb59(34)
      abb59(13)=abb59(32)*abb59(13)
      abb59(12)=abb59(30)*abb59(12)*mT
      abb59(35)=-abb59(13)-abb59(12)
      abb59(35)=c1*abb59(35)
      abb59(34)=abb59(34)+abb59(35)
      abb59(35)=spak2l3*spbl3k2
      abb59(34)=abb59(35)*abb59(1)*abb59(34)
      abb59(36)=spak2l5*spbl5k2
      abb59(37)=-abb59(19)*abb59(36)
      abb59(38)=spbl4k2*abb59(26)*abb59(1)**3
      abb59(34)=abb59(37)+abb59(38)+abb59(34)
      abb59(34)=spak1l5*abb59(34)
      abb59(37)=abb59(24)*mT
      abb59(25)=abb59(25)*abb59(17)
      abb59(25)=abb59(37)-abb59(25)
      abb59(18)=-abb59(18)*abb59(25)
      abb59(29)=abb59(29)*abb59(5)
      abb59(37)=abb59(18)*abb59(29)
      abb59(32)=abb59(1)*abb59(32)*abb59(14)
      abb59(30)=c2*abb59(33)*abb59(30)
      abb59(30)=abb59(32)-abb59(30)
      abb59(32)=abb59(5)*spbl4k2
      abb59(33)=abb59(30)*abb59(32)
      abb59(38)=-abb59(35)*abb59(33)
      abb59(37)=abb59(37)+abb59(38)
      abb59(37)=spak1l4*abb59(37)
      abb59(38)=abb59(32)*spak1l3
      abb59(39)=-spbl3k2*abb59(18)*abb59(38)
      abb59(21)=abb59(21)*abb59(27)
      abb59(40)=spbl4l3*spak1l3
      abb59(41)=-abb59(21)*abb59(40)
      abb59(23)=abb59(41)+abb59(37)+abb59(39)+abb59(34)+abb59(23)
      abb59(23)=2.0_ki*abb59(23)
      abb59(34)=abb59(15)*spak1l5
      abb59(37)=4.0_ki*abb59(34)
      abb59(39)=-abb59(35)*abb59(37)
      abb59(38)=abb59(38)*abb59(25)
      abb59(41)=abb59(38)*spbl3k2
      abb59(42)=4.0_ki*abb59(41)
      abb59(11)=-2.0_ki*abb59(11)-3.0_ki*abb59(31)
      abb59(11)=c2*abb59(11)
      abb59(12)=2.0_ki*abb59(13)+3.0_ki*abb59(12)
      abb59(12)=c1*abb59(12)
      abb59(11)=abb59(11)+abb59(12)
      abb59(11)=spak1l5*abb59(1)*abb59(11)
      abb59(12)=abb59(2)*abb59(16)
      abb59(12)=abb59(12)-abb59(14)
      abb59(13)=mT**3
      abb59(12)=abb59(5)*abb59(9)*abb59(12)*abb59(13)
      abb59(13)=-abb59(8)*abb59(26)*abb59(13)*abb59(4)**2
      abb59(12)=abb59(12)-abb59(13)
      abb59(13)=spak1k2*abb59(20)*abb59(12)
      abb59(14)=spak1l4*abb59(33)
      abb59(11)=abb59(11)+2.0_ki*abb59(14)+abb59(13)
      abb59(11)=4.0_ki*abb59(11)
      abb59(13)=spak1k2*abb59(22)
      abb59(14)=abb59(36)-abb59(35)
      abb59(14)=spak1l5*abb59(14)
      abb59(13)=abb59(14)+abb59(13)
      abb59(13)=abb59(15)*abb59(13)
      abb59(14)=abb59(29)*abb59(25)
      abb59(16)=-spak1l4*abb59(14)
      abb59(22)=abb59(27)*spak1l5
      abb59(26)=spbl4k2*abb59(22)
      abb59(13)=abb59(16)+abb59(41)+abb59(26)+abb59(13)
      abb59(13)=2.0_ki*abb59(13)
      abb59(16)=spak1l3*spbl3k2*abb59(25)
      abb59(26)=-spak1l4*abb59(15)*abb59(35)
      abb59(16)=abb59(16)+abb59(26)
      abb59(16)=2.0_ki*abb59(16)
      abb59(12)=abb59(35)*abb59(12)
      abb59(26)=abb59(1)-mT
      abb59(24)=abb59(26)*abb59(24)
      abb59(26)=abb59(2)*abb59(1)
      abb59(17)=abb59(26)-abb59(17)
      abb59(10)=c2*abb59(10)*abb59(17)
      abb59(10)=abb59(24)-abb59(10)
      abb59(17)=-spbl4k2*abb59(10)*abb59(28)
      abb59(12)=3.0_ki*abb59(19)+abb59(17)+abb59(12)
      abb59(12)=spak1k2*abb59(12)
      abb59(10)=-abb59(10)*abb59(40)
      abb59(10)=abb59(10)+abb59(12)
      abb59(10)=2.0_ki*abb59(10)
      abb59(12)=2.0_ki*abb59(15)
      abb59(15)=-spak1k2*abb59(12)
      abb59(14)=spak1k2*abb59(28)*abb59(14)
      abb59(17)=spbl4l3*abb59(38)
      abb59(14)=abb59(14)+abb59(17)
      abb59(14)=2.0_ki*abb59(14)
      abb59(17)=-2.0_ki*spbl3k2*abb59(22)
      abb59(19)=-2.0_ki*abb59(20)*abb59(34)
      abb59(12)=-spak1l4*abb59(20)*abb59(12)
      abb59(20)=-4.0_ki*abb59(21)
      abb59(21)=4.0_ki*abb59(27)
      abb59(18)=abb59(18)*abb59(32)
      abb59(22)=abb59(30)*abb59(5)
      abb59(24)=-abb59(35)*abb59(22)
      abb59(18)=3.0_ki*abb59(18)+abb59(24)
      abb59(18)=2.0_ki*abb59(18)
      abb59(22)=8.0_ki*abb59(22)
      abb59(24)=2.0_ki*abb59(32)
      abb59(24)=-abb59(25)*abb59(24)
      R2d59=0.0_ki
      rat2 = rat2 + R2d59
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='59' value='", &
          & R2d59, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd59h5
