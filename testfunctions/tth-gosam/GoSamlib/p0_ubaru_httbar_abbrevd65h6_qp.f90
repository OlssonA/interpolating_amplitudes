module     p0_ubaru_httbar_abbrevd65h6_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh6_qp
   implicit none
   private
   complex(ki), dimension(43), public :: abb65
   complex(ki), public :: R2d65
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
      abb65(1)=1.0_ki/(-mT**2+es34)
      abb65(2)=sqrt(mT**2)
      abb65(3)=NC**(-1)
      abb65(4)=spak2l3**(-1)
      abb65(5)=spbl3k2**(-1)
      abb65(6)=spbl5k2**(-1)
      abb65(7)=spak2l4**(-1)
      abb65(8)=c2*spak2l5
      abb65(9)=i_*e*gHT*abb65(1)*TR**2*gs**4
      abb65(10)=abb65(8)*abb65(9)
      abb65(11)=abb65(10)*abb65(2)
      abb65(12)=-NC+2.0_ki*abb65(3)
      abb65(13)=-abb65(11)*abb65(12)
      abb65(14)=abb65(3)**2
      abb65(15)=abb65(14)*spak2l5
      abb65(16)=abb65(15)*c1
      abb65(17)=abb65(9)*abb65(2)
      abb65(18)=abb65(16)*abb65(17)
      abb65(13)=abb65(18)+abb65(13)
      abb65(18)=4.0_ki*spbl4k1*abb65(13)
      abb65(19)=abb65(10)*abb65(3)
      abb65(20)=-NC*abb65(10)
      abb65(20)=abb65(20)+2.0_ki*abb65(19)
      abb65(21)=abb65(2)*abb65(7)
      abb65(22)=mT**2
      abb65(23)=abb65(21)*abb65(22)
      abb65(24)=mT**3
      abb65(25)=abb65(24)*abb65(7)
      abb65(25)=abb65(25)+abb65(23)
      abb65(20)=abb65(20)*abb65(25)*abb65(6)
      abb65(25)=abb65(9)*abb65(6)
      abb65(26)=abb65(23)*abb65(25)
      abb65(27)=abb65(24)*abb65(25)
      abb65(28)=-abb65(7)*abb65(27)
      abb65(28)=abb65(28)-abb65(26)
      abb65(28)=abb65(28)*abb65(16)
      abb65(29)=abb65(8)*NC
      abb65(30)=abb65(8)*abb65(3)
      abb65(16)=-abb65(16)-abb65(29)+2.0_ki*abb65(30)
      abb65(29)=abb65(9)*mT
      abb65(31)=abb65(29)*abb65(6)
      abb65(16)=abb65(16)*abb65(31)
      abb65(32)=spbl4k2*abb65(4)*abb65(5)*mH**2
      abb65(33)=abb65(16)*abb65(32)
      abb65(20)=abb65(33)+abb65(28)+abb65(20)
      abb65(20)=spbl5k1*abb65(20)
      abb65(9)=abb65(15)*abb65(9)
      abb65(28)=abb65(14)*abb65(31)
      abb65(33)=-abb65(2)*abb65(28)
      abb65(33)=-abb65(9)+abb65(33)
      abb65(33)=c1*abb65(33)
      abb65(31)=abb65(31)*c2
      abb65(34)=abb65(31)*abb65(2)
      abb65(35)=abb65(3)*abb65(34)
      abb65(35)=abb65(19)+abb65(35)
      abb65(34)=-abb65(10)-abb65(34)
      abb65(34)=NC*abb65(34)
      abb65(33)=abb65(34)+2.0_ki*abb65(35)+abb65(33)
      abb65(33)=spbl4k1*abb65(2)*abb65(33)
      abb65(34)=-c1*abb65(14)*abb65(25)
      abb65(25)=abb65(25)*c2
      abb65(35)=abb65(25)*abb65(3)
      abb65(36)=-NC*abb65(25)
      abb65(34)=abb65(36)+2.0_ki*abb65(35)+abb65(34)
      abb65(34)=spbl3k1*spak2l3*abb65(23)*abb65(34)
      abb65(20)=abb65(34)+abb65(33)+abb65(20)
      abb65(20)=4.0_ki*abb65(20)
      abb65(33)=abb65(11)*abb65(3)
      abb65(30)=abb65(30)*abb65(29)
      abb65(33)=-abb65(30)-abb65(33)
      abb65(34)=abb65(15)*abb65(29)
      abb65(15)=abb65(17)*abb65(15)
      abb65(15)=abb65(34)+abb65(15)
      abb65(15)=c1*abb65(15)
      abb65(8)=abb65(29)*abb65(8)
      abb65(11)=abb65(8)+abb65(11)
      abb65(11)=NC*abb65(11)
      abb65(11)=abb65(11)+2.0_ki*abb65(33)+abb65(15)
      abb65(11)=spbl4k1*abb65(11)
      abb65(15)=abb65(8)*NC
      abb65(17)=abb65(34)*c1
      abb65(15)=-abb65(15)-abb65(17)+2.0_ki*abb65(30)
      abb65(17)=spak2l3*abb65(7)*abb65(15)
      abb65(29)=-spbl3k1*abb65(17)
      abb65(11)=abb65(11)+abb65(29)
      abb65(11)=4.0_ki*abb65(11)
      abb65(29)=2.0_ki*spbl5k1
      abb65(33)=-abb65(15)*abb65(29)
      abb65(29)=-abb65(29)*abb65(17)
      abb65(15)=spbl5l4*abb65(15)
      abb65(17)=spbl5l3*abb65(17)
      abb65(15)=abb65(15)+abb65(17)
      abb65(15)=2.0_ki*abb65(15)
      abb65(17)=2.0_ki*spbl4l3
      abb65(16)=spbl5k1*abb65(16)*abb65(17)
      abb65(36)=abb65(13)*abb65(17)
      abb65(24)=abb65(25)*abb65(24)
      abb65(37)=abb65(24)-abb65(8)
      abb65(37)=abb65(37)*NC
      abb65(27)=abb65(27)*abb65(14)
      abb65(38)=abb65(27)-abb65(34)
      abb65(38)=abb65(38)*c1
      abb65(37)=abb65(37)+abb65(38)
      abb65(38)=abb65(24)*abb65(3)
      abb65(39)=abb65(38)-abb65(30)
      abb65(40)=2.0_ki*abb65(39)-abb65(37)
      abb65(41)=2.0_ki*spbk2k1
      abb65(42)=-abb65(40)*abb65(41)
      abb65(37)=-abb65(7)*abb65(37)
      abb65(43)=2.0_ki*abb65(7)
      abb65(39)=abb65(39)*abb65(43)
      abb65(37)=abb65(39)+abb65(37)
      abb65(37)=abb65(37)*spak2l3
      abb65(39)=-abb65(41)*abb65(37)
      abb65(22)=abb65(22)*abb65(7)
      abb65(19)=-abb65(22)*abb65(19)
      abb65(30)=-abb65(30)*abb65(21)
      abb65(19)=abb65(19)+abb65(30)
      abb65(9)=abb65(22)*abb65(9)
      abb65(30)=abb65(21)*abb65(34)
      abb65(9)=abb65(9)+abb65(30)
      abb65(9)=c1*abb65(9)
      abb65(10)=abb65(10)*abb65(22)
      abb65(8)=abb65(8)*abb65(21)
      abb65(8)=abb65(10)+abb65(8)
      abb65(8)=NC*abb65(8)
      abb65(8)=abb65(8)+2.0_ki*abb65(19)+abb65(9)
      abb65(8)=abb65(2)*abb65(8)
      abb65(9)=-spbl4k2*abb65(40)
      abb65(10)=-spbl3k2*abb65(37)
      abb65(13)=abb65(13)*abb65(32)
      abb65(8)=abb65(10)+abb65(13)+abb65(9)+abb65(8)
      abb65(8)=2.0_ki*abb65(8)
      abb65(9)=abb65(7)*abb65(38)
      abb65(10)=abb65(35)*abb65(23)
      abb65(9)=abb65(9)+abb65(10)
      abb65(10)=-abb65(7)*abb65(27)
      abb65(13)=-abb65(14)*abb65(26)
      abb65(10)=abb65(10)+abb65(13)
      abb65(10)=c1*abb65(10)
      abb65(13)=-abb65(7)*abb65(24)
      abb65(14)=-abb65(25)*abb65(23)
      abb65(13)=abb65(13)+abb65(14)
      abb65(13)=NC*abb65(13)
      abb65(12)=abb65(31)*abb65(12)
      abb65(14)=abb65(28)*c1
      abb65(12)=-abb65(14)+abb65(12)
      abb65(14)=abb65(12)*abb65(32)
      abb65(9)=abb65(14)+abb65(13)+2.0_ki*abb65(9)+abb65(10)
      abb65(9)=4.0_ki*abb65(9)
      abb65(10)=abb65(12)*abb65(17)
      R2d65=0.0_ki
      rat2 = rat2 + R2d65
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='65' value='", &
          & R2d65, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd65h6_qp
