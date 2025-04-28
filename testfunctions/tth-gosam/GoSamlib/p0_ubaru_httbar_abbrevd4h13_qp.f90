module     p0_ubaru_httbar_abbrevd4h13_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh13_qp
   implicit none
   private
   complex(ki), dimension(51), public :: abb4
   complex(ki), public :: R2d4
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p0_ubaru_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_color_qp, only: TR
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      implicit none
      abb4(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb4(2)=es12**(-1)
      abb4(3)=spak2l3**(-1)
      abb4(4)=spbl3k2**(-1)
      abb4(5)=spak2l4**(-1)
      abb4(6)=spak2l5**(-1)
      abb4(7)=sqrt(mT**2)
      abb4(8)=spbl4k2**(-1)
      abb4(9)=NC*c2
      abb4(10)=c1-abb4(9)
      abb4(11)=i_*e*gHT*abb4(1)*TR**2*gs**4
      abb4(10)=abb4(10)*abb4(11)*spak1k2*abb4(2)
      abb4(12)=spbl5k2*mH**2*abb4(4)*abb4(3)
      abb4(13)=abb4(10)*abb4(12)
      abb4(14)=abb4(13)*spbl4k2
      abb4(15)=mT**2
      abb4(16)=-abb4(15)*abb4(10)
      abb4(17)=abb4(6)*spbl4k2
      abb4(18)=abb4(16)*abb4(17)
      abb4(19)=-mT*abb4(10)
      abb4(20)=abb4(17)*abb4(7)
      abb4(21)=abb4(20)*abb4(19)
      abb4(14)=-abb4(14)+abb4(18)+abb4(21)
      abb4(9)=abb4(11)*abb4(9)
      abb4(11)=abb4(11)*c1
      abb4(22)=abb4(9)-abb4(11)
      abb4(23)=abb4(2)*abb4(22)
      abb4(24)=abb4(23)*spbl5l3
      abb4(25)=abb4(24)*spak1l3
      abb4(26)=abb4(25)*spbl4k2
      abb4(26)=abb4(26)+abb4(14)
      abb4(27)=abb4(7)*abb4(5)
      abb4(28)=abb4(19)*abb4(27)
      abb4(29)=abb4(16)*abb4(5)
      abb4(30)=abb4(28)+abb4(29)
      abb4(31)=abb4(30)*spbl5k2
      abb4(32)=spak2l3*abb4(6)
      abb4(33)=abb4(32)*spbl3k2
      abb4(34)=abb4(33)*abb4(29)
      abb4(35)=abb4(31)+abb4(34)
      abb4(36)=abb4(26)+abb4(35)
      abb4(37)=abb4(16)*abb4(6)
      abb4(38)=abb4(7)*abb4(6)
      abb4(39)=abb4(19)*abb4(38)
      abb4(13)=abb4(25)-abb4(13)+abb4(37)+abb4(39)
      abb4(25)=spbl4k2**2
      abb4(37)=spak2l4*abb4(25)*abb4(13)
      abb4(39)=abb4(9)*spak1k2
      abb4(40)=abb4(11)*spak1k2
      abb4(39)=abb4(39)-abb4(40)
      abb4(41)=mT**4
      abb4(42)=-abb4(41)*abb4(39)
      abb4(43)=mT**3
      abb4(44)=abb4(39)*abb4(43)
      abb4(45)=abb4(7)*abb4(44)
      abb4(45)=-abb4(42)+abb4(45)
      abb4(46)=abb4(5)**2
      abb4(47)=abb4(46)*spbl5k2
      abb4(45)=abb4(45)*abb4(47)
      abb4(46)=abb4(46)*abb4(33)
      abb4(48)=-abb4(42)*abb4(46)
      abb4(45)=abb4(45)+abb4(48)
      abb4(45)=abb4(8)*abb4(45)
      abb4(37)=abb4(45)+abb4(37)
      abb4(31)=2.0_ki*abb4(31)+2.0_ki*abb4(34)
      abb4(14)=abb4(31)+abb4(14)
      abb4(14)=spbl4k1*abb4(14)
      abb4(34)=abb4(30)*spbl4k2
      abb4(45)=-spbl5k1*abb4(34)
      abb4(14)=abb4(45)+abb4(14)
      abb4(14)=spak1l4*abb4(14)
      abb4(18)=-abb4(18)+3.0_ki*abb4(21)
      abb4(18)=spbl3k2*abb4(18)
      abb4(21)=abb4(24)*spak1l4
      abb4(25)=-abb4(25)*abb4(21)
      abb4(45)=abb4(29)*abb4(17)
      abb4(48)=-spbl3k1*spak1l4*abb4(45)
      abb4(18)=abb4(48)+abb4(18)+abb4(25)
      abb4(18)=spak2l3*abb4(18)
      abb4(25)=abb4(5)*abb4(44)
      abb4(39)=abb4(39)*abb4(15)
      abb4(44)=3.0_ki*abb4(27)
      abb4(48)=abb4(39)*abb4(44)
      abb4(25)=2.0_ki*abb4(25)+abb4(48)
      abb4(25)=abb4(25)*abb4(38)
      abb4(48)=2.0_ki*spbl4k2
      abb4(19)=abb4(48)*abb4(19)
      abb4(49)=abb4(10)*spbl4k2
      abb4(50)=abb4(7)*abb4(49)
      abb4(50)=abb4(19)-3.0_ki*abb4(50)
      abb4(50)=abb4(7)*abb4(50)
      abb4(51)=-spbl4k2*abb4(16)
      abb4(50)=abb4(51)+abb4(50)
      abb4(50)=spbl5k2*abb4(50)
      abb4(39)=-abb4(5)*abb4(39)
      abb4(9)=abb4(9)*mT
      abb4(51)=spak1k2*abb4(9)
      abb4(40)=-mT*abb4(40)
      abb4(40)=abb4(40)+abb4(51)
      abb4(40)=abb4(40)*abb4(44)
      abb4(39)=abb4(39)+abb4(40)
      abb4(39)=abb4(39)*abb4(12)
      abb4(22)=-abb4(5)*abb4(15)*abb4(22)
      abb4(11)=-mT*abb4(11)
      abb4(9)=abb4(11)+abb4(9)
      abb4(9)=abb4(9)*abb4(44)
      abb4(9)=abb4(22)+abb4(9)
      abb4(9)=spbl5l3*abb4(9)
      abb4(11)=abb4(21)*spbl4k2
      abb4(21)=spbl4k1*abb4(11)
      abb4(9)=abb4(9)+abb4(21)
      abb4(9)=spak1l3*abb4(9)
      abb4(21)=abb4(6)*abb4(5)*abb4(42)
      abb4(9)=abb4(9)+abb4(39)+abb4(50)+abb4(21)+abb4(25)+abb4(18)+2.0_ki*abb4(&
      &37)+abb4(14)
      abb4(14)=2.0_ki*abb4(26)
      abb4(18)=-abb4(41)*abb4(10)
      abb4(10)=abb4(10)*abb4(43)
      abb4(21)=-abb4(7)*abb4(10)
      abb4(21)=abb4(18)+abb4(21)
      abb4(21)=abb4(21)*abb4(47)
      abb4(22)=abb4(18)*abb4(46)
      abb4(21)=abb4(21)+abb4(22)
      abb4(21)=abb4(8)*abb4(21)
      abb4(18)=abb4(5)*abb4(18)
      abb4(16)=abb4(16)*abb4(48)
      abb4(16)=abb4(18)+abb4(16)
      abb4(16)=abb4(6)*abb4(16)
      abb4(10)=-abb4(5)*abb4(10)
      abb4(10)=abb4(10)+abb4(19)
      abb4(10)=abb4(10)*abb4(38)
      abb4(18)=abb4(29)-2.0_ki*abb4(49)
      abb4(18)=abb4(18)*abb4(12)
      abb4(19)=-mT*abb4(23)
      abb4(22)=abb4(19)*abb4(27)
      abb4(25)=abb4(23)*spbl4k2
      abb4(27)=2.0_ki*abb4(25)-abb4(22)
      abb4(27)=spak1l3*spbl5l3*abb4(27)
      abb4(10)=abb4(21)+abb4(27)+abb4(18)+abb4(16)+abb4(10)-abb4(35)
      abb4(10)=2.0_ki*abb4(10)
      abb4(16)=-abb4(26)+abb4(31)
      abb4(16)=2.0_ki*abb4(16)
      abb4(18)=2.0_ki*abb4(36)
      abb4(21)=spak2l3*abb4(45)
      abb4(26)=abb4(29)-abb4(28)
      abb4(26)=spbl5l3*abb4(26)
      abb4(15)=-abb4(15)*abb4(23)
      abb4(27)=abb4(15)*abb4(5)
      abb4(28)=abb4(22)+abb4(27)
      abb4(31)=-spbl5k2*abb4(28)
      abb4(20)=abb4(20)*abb4(19)
      abb4(17)=-abb4(15)*abb4(17)
      abb4(35)=abb4(25)*abb4(12)
      abb4(33)=-abb4(33)*abb4(27)
      abb4(17)=abb4(33)+abb4(35)+abb4(31)+abb4(17)-abb4(20)
      abb4(17)=spak1l4*abb4(17)
      abb4(31)=-spbl4k2*abb4(19)
      abb4(33)=abb4(7)*abb4(25)
      abb4(31)=abb4(31)+abb4(33)
      abb4(31)=abb4(7)*abb4(31)
      abb4(33)=abb4(23)*abb4(7)
      abb4(35)=-abb4(33)+abb4(19)
      abb4(35)=spbl5k2*abb4(7)*abb4(35)
      abb4(37)=abb4(19)*abb4(38)
      abb4(38)=spbl3k2*spak2l3*abb4(37)
      abb4(35)=abb4(35)+abb4(38)
      abb4(20)=-spak2l3*abb4(20)
      abb4(13)=2.0_ki*abb4(13)
      abb4(13)=spbl4k1*abb4(13)
      abb4(27)=2.0_ki*abb4(27)
      abb4(22)=4.0_ki*abb4(22)-abb4(27)-abb4(25)
      abb4(22)=spbl5l3*abb4(22)
      abb4(25)=abb4(15)*abb4(6)
      abb4(38)=-abb4(25)+2.0_ki*abb4(37)
      abb4(38)=spbl4l3*abb4(38)
      abb4(22)=2.0_ki*abb4(38)+abb4(22)
      abb4(22)=spak2l3*abb4(22)
      abb4(30)=-spbl5k1*abb4(30)
      abb4(19)=2.0_ki*abb4(33)-abb4(19)
      abb4(19)=abb4(7)*abb4(19)
      abb4(15)=abb4(19)+abb4(15)
      abb4(15)=spbl5l4*abb4(15)
      abb4(19)=-abb4(32)*abb4(29)*spbl3k1
      abb4(13)=abb4(19)+2.0_ki*abb4(15)+abb4(30)+abb4(13)+abb4(22)
      abb4(15)=2.0_ki*abb4(24)
      abb4(19)=-2.0_ki*abb4(28)
      abb4(12)=abb4(23)*abb4(12)
      abb4(12)=abb4(12)-abb4(25)-abb4(37)
      abb4(12)=2.0_ki*abb4(12)
      abb4(22)=-abb4(32)*abb4(27)
      R2d4=-abb4(36)
      rat2 = rat2 + R2d4
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='4' value='", &
          & R2d4, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd4h13_qp
