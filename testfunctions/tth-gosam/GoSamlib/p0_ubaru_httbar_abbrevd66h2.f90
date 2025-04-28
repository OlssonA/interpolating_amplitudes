module     p0_ubaru_httbar_abbrevd66h2
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_kinematics, only: epstensor
   use p0_ubaru_httbar_globalsh2
   implicit none
   private
   complex(ki), dimension(42), public :: abb66
   complex(ki), public :: R2d66
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
      abb66(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb66(2)=NC**(-1)
      abb66(3)=spak2l4**(-1)
      abb66(4)=spbl4k2**(-1)
      abb66(5)=spbl5k2**(-1)
      abb66(6)=sqrt(mT**2)
      abb66(7)=spak2l3**(-1)
      abb66(8)=spbl3k2**(-1)
      abb66(9)=mT**3
      abb66(10)=i_*e*gHT*abb66(1)*TR**2*gs**4
      abb66(11)=abb66(10)*c2
      abb66(12)=abb66(11)*abb66(2)
      abb66(13)=spbk2k1*abb66(4)
      abb66(14)=abb66(9)*abb66(12)*abb66(13)
      abb66(15)=mT**2
      abb66(16)=abb66(15)*abb66(13)
      abb66(17)=abb66(16)*abb66(12)
      abb66(18)=abb66(6)*abb66(17)
      abb66(14)=abb66(14)+abb66(18)
      abb66(18)=2.0_ki*abb66(5)
      abb66(14)=abb66(14)*abb66(18)
      abb66(19)=abb66(2)**2
      abb66(20)=abb66(19)*abb66(10)
      abb66(21)=abb66(13)*abb66(20)
      abb66(22)=-abb66(9)*abb66(21)
      abb66(23)=abb66(21)*abb66(15)
      abb66(24)=-abb66(6)*abb66(23)
      abb66(22)=abb66(22)+abb66(24)
      abb66(24)=c1*abb66(5)
      abb66(22)=abb66(22)*abb66(24)
      abb66(25)=abb66(11)*abb66(13)
      abb66(26)=-abb66(9)*abb66(25)
      abb66(27)=abb66(25)*abb66(15)
      abb66(28)=-abb66(6)*abb66(27)
      abb66(26)=abb66(26)+abb66(28)
      abb66(28)=NC*abb66(5)
      abb66(26)=abb66(26)*abb66(28)
      abb66(14)=abb66(26)+abb66(14)+abb66(22)
      abb66(14)=abb66(6)*abb66(14)
      abb66(22)=2.0_ki*abb66(12)
      abb66(26)=abb66(11)*NC
      abb66(29)=abb66(22)-abb66(26)
      abb66(30)=spbk2k1*abb66(4)**2
      abb66(9)=abb66(30)*abb66(9)
      abb66(31)=abb66(9)*abb66(6)
      abb66(30)=abb66(30)*mT**4
      abb66(31)=abb66(31)+abb66(30)
      abb66(31)=abb66(31)*abb66(29)
      abb66(32)=abb66(10)*abb66(6)
      abb66(33)=abb66(19)*abb66(32)
      abb66(9)=-abb66(33)*abb66(9)
      abb66(34)=-abb66(30)*abb66(20)
      abb66(9)=abb66(34)+abb66(9)
      abb66(9)=c1*abb66(9)
      abb66(9)=abb66(9)+abb66(31)
      abb66(9)=spak2l5*abb66(9)
      abb66(31)=abb66(19)*c1
      abb66(34)=abb66(31)*abb66(10)
      abb66(26)=abb66(26)+abb66(34)
      abb66(34)=-abb66(5)*abb66(26)
      abb66(35)=abb66(22)*abb66(5)
      abb66(34)=abb66(35)+abb66(34)
      abb66(36)=spak2l3*spbl3k2
      abb66(30)=abb66(36)*abb66(30)*abb66(34)
      abb66(9)=abb66(30)+abb66(9)
      abb66(9)=abb66(3)*abb66(9)
      abb66(30)=abb66(6)*mT
      abb66(34)=abb66(13)*abb66(22)*abb66(30)
      abb66(37)=abb66(21)*abb66(30)
      abb66(38)=abb66(37)*c1
      abb66(39)=abb66(25)*abb66(30)
      abb66(40)=abb66(39)*NC
      abb66(34)=abb66(34)-abb66(38)-abb66(40)
      abb66(38)=spak2l5*abb66(7)*abb66(8)*mH**2
      abb66(40)=abb66(34)*abb66(38)
      abb66(9)=abb66(40)+abb66(14)+abb66(9)
      abb66(9)=4.0_ki*abb66(9)
      abb66(14)=abb66(12)*abb66(30)
      abb66(13)=abb66(14)*abb66(13)
      abb66(13)=abb66(13)+abb66(17)
      abb66(17)=abb66(27)+abb66(39)
      abb66(17)=abb66(17)*NC
      abb66(23)=abb66(23)+abb66(37)
      abb66(23)=abb66(23)*c1
      abb66(13)=-2.0_ki*abb66(13)+abb66(17)+abb66(23)
      abb66(17)=-spak2l5*abb66(13)
      abb66(21)=abb66(21)*c1
      abb66(23)=abb66(25)*NC
      abb66(21)=abb66(23)+abb66(21)
      abb66(23)=abb66(15)*abb66(5)
      abb66(21)=abb66(23)*abb66(21)
      abb66(16)=abb66(35)*abb66(16)
      abb66(16)=-abb66(16)+abb66(21)
      abb66(21)=-abb66(16)*abb66(36)
      abb66(17)=abb66(17)+abb66(21)
      abb66(17)=4.0_ki*abb66(17)
      abb66(21)=2.0_ki*spak2l4
      abb66(25)=-abb66(13)*abb66(21)
      abb66(12)=abb66(12)*abb66(15)
      abb66(12)=abb66(12)+abb66(14)
      abb66(10)=abb66(15)*abb66(10)
      abb66(14)=abb66(19)*abb66(10)
      abb66(19)=abb66(33)*mT
      abb66(14)=abb66(14)+abb66(19)
      abb66(19)=-c1*abb66(14)
      abb66(27)=abb66(30)*abb66(11)
      abb66(11)=abb66(11)*abb66(15)
      abb66(15)=abb66(27)+abb66(11)
      abb66(37)=-NC*abb66(15)
      abb66(19)=abb66(37)+2.0_ki*abb66(12)+abb66(19)
      abb66(37)=2.0_ki*es12
      abb66(19)=abb66(37)*abb66(4)*abb66(19)
      abb66(26)=-abb66(22)+abb66(26)
      abb66(39)=2.0_ki*spal3l5
      abb66(40)=abb66(39)*abb66(26)
      abb66(41)=spbl4k1*spak2l4
      abb66(42)=-abb66(41)*abb66(40)
      abb66(14)=abb66(14)*abb66(24)
      abb66(12)=abb66(12)*abb66(18)
      abb66(15)=abb66(15)*abb66(28)
      abb66(18)=abb66(38)*abb66(26)
      abb66(12)=abb66(18)+abb66(15)+abb66(14)-abb66(12)
      abb66(12)=2.0_ki*abb66(12)
      abb66(14)=-abb66(41)*abb66(12)
      abb66(15)=abb66(30)*abb66(35)
      abb66(18)=abb66(31)*abb66(5)
      abb66(24)=-mT*abb66(32)*abb66(18)
      abb66(26)=-abb66(27)*abb66(28)
      abb66(15)=abb66(26)+abb66(15)+abb66(24)
      abb66(15)=abb66(15)*abb66(36)
      abb66(20)=-mT*abb66(20)
      abb66(20)=abb66(20)-abb66(33)
      abb66(20)=c1*abb66(20)
      abb66(24)=abb66(6)+mT
      abb66(24)=abb66(24)*abb66(29)
      abb66(20)=abb66(20)+abb66(24)
      abb66(20)=spak2l5*abb66(6)*abb66(20)
      abb66(15)=abb66(20)+abb66(15)
      abb66(15)=2.0_ki*abb66(15)
      abb66(16)=abb66(16)*spbl3k2
      abb66(20)=-abb66(16)*abb66(21)
      abb66(21)=abb66(23)*abb66(22)
      abb66(10)=-abb66(10)*abb66(18)
      abb66(11)=-abb66(11)*abb66(28)
      abb66(10)=abb66(11)+abb66(21)+abb66(10)
      abb66(10)=spbl3k2*abb66(37)*abb66(4)*abb66(10)
      abb66(11)=-spal4l5*abb66(13)
      abb66(18)=spal3l4*abb66(16)
      abb66(11)=abb66(11)+abb66(18)
      abb66(11)=2.0_ki*abb66(11)
      abb66(18)=abb66(34)*abb66(39)
      abb66(13)=spak1l5*abb66(13)
      abb66(16)=spak1l3*abb66(16)
      abb66(13)=abb66(13)+abb66(16)
      abb66(13)=2.0_ki*abb66(13)
      R2d66=0.0_ki
      rat2 = rat2 + R2d66
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='66' value='", &
          & R2d66, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd66h2
