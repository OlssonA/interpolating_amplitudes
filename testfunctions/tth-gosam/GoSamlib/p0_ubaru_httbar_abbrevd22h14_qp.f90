module     p0_ubaru_httbar_abbrevd22h14_qp
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_kinematics_qp, only: epstensor
   use p0_ubaru_httbar_globalsh14_qp
   implicit none
   private
   complex(ki), dimension(30), public :: abb22
   complex(ki), public :: R2d22
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
      abb22(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb22(2)=1.0_ki/(mH**2+2.0_ki*mT**2-es34-es45+es12-3.0_ki*sqrt(mT**2)**2)
      abb22(3)=sqrt(mT**2)
      abb22(4)=NC**(-1)
      abb22(5)=es12**(-1)
      abb22(6)=spbl3k2**(-1)
      abb22(7)=spak2l5**(-1)
      abb22(8)=spbl5k2**(-1)
      abb22(9)=1.0_ki/(sqrt(mT**2)**2)
      abb22(10)=spak2l3**(-1)
      abb22(11)=es34+es45
      abb22(12)=mH**2
      abb22(13)=abb22(11)-abb22(12)
      abb22(14)=-abb22(2)*abb22(13)
      abb22(15)=mT**2
      abb22(16)=abb22(15)*abb22(2)
      abb22(16)=abb22(16)-abb22(14)
      abb22(17)=abb22(4)*c1
      abb22(18)=abb22(17)-c2
      abb22(18)=abb22(18)*abb22(4)
      abb22(19)=abb22(18)-c1
      abb22(16)=abb22(16)*abb22(19)
      abb22(14)=-NC*abb22(14)
      abb22(20)=abb22(15)*NC
      abb22(21)=abb22(2)*abb22(20)
      abb22(14)=abb22(21)+abb22(14)
      abb22(14)=c2*abb22(14)
      abb22(14)=abb22(14)+abb22(16)
      abb22(14)=abb22(5)*abb22(14)
      abb22(16)=c2*NC
      abb22(18)=abb22(18)+abb22(16)-c1
      abb22(21)=-abb22(2)*abb22(18)
      abb22(14)=abb22(14)+abb22(21)
      abb22(21)=spbl5l3*spak2l3
      abb22(22)=abb22(3)**2
      abb22(23)=abb22(22)*deltaOS
      abb22(24)=abb22(21)*abb22(23)
      abb22(25)=abb22(1)*TR
      abb22(25)=i_*gs**4*gHT*e*spbl4k1*abb22(25)**2
      abb22(14)=4.0_ki*abb22(25)*abb22(24)*abb22(14)
      abb22(17)=abb22(17)*abb22(9)
      abb22(26)=c2*abb22(9)
      abb22(17)=abb22(17)-abb22(26)
      abb22(17)=abb22(17)*abb22(4)
      abb22(27)=c1*abb22(9)
      abb22(28)=abb22(17)-abb22(27)
      abb22(13)=2.0_ki*abb22(15)-abb22(13)
      abb22(13)=abb22(13)*abb22(28)
      abb22(28)=abb22(12)*NC
      abb22(11)=-NC*abb22(11)
      abb22(11)=abb22(28)+abb22(11)+2.0_ki*abb22(20)
      abb22(11)=abb22(11)*abb22(26)
      abb22(11)=abb22(11)+abb22(13)
      abb22(11)=deltaOS*abb22(11)
      abb22(13)=abb22(16)*abb22(9)
      abb22(13)=abb22(17)+abb22(13)-abb22(27)
      abb22(16)=abb22(13)*abb22(23)
      abb22(11)=-abb22(16)+abb22(11)+abb22(18)
      abb22(11)=abb22(11)*abb22(22)
      abb22(17)=abb22(15)+abb22(12)
      abb22(17)=-abb22(17)*abb22(19)
      abb22(26)=-abb22(20)-abb22(28)
      abb22(26)=c2*abb22(26)
      abb22(27)=abb22(15)*abb22(7)
      abb22(28)=abb22(27)*abb22(8)
      abb22(29)=-spak2l3*abb22(18)*abb22(28)*spbl3k2
      abb22(11)=abb22(29)+3.0_ki*abb22(11)+abb22(26)+abb22(17)
      abb22(11)=spak2l3*abb22(11)
      abb22(12)=abb22(12)*abb22(6)
      abb22(17)=spbl5k2*abb22(12)
      abb22(26)=abb22(17)*spak2l5
      abb22(29)=-abb22(26)*abb22(18)
      abb22(30)=-abb22(21)*spal3l5*abb22(18)
      abb22(11)=abb22(30)+abb22(11)+abb22(29)
      abb22(11)=abb22(5)*spbl5l3*abb22(11)
      abb22(24)=abb22(13)*abb22(24)
      abb22(11)=3.0_ki*abb22(24)+abb22(11)
      abb22(11)=2.0_ki*abb22(11)*abb22(25)
      abb22(25)=abb22(5)*abb22(25)
      abb22(29)=12.0_ki*abb22(25)
      abb22(24)=abb22(29)*abb22(24)
      abb22(16)=3.0_ki*abb22(16)-abb22(18)
      abb22(16)=4.0_ki*abb22(16)*abb22(25)*abb22(21)
      abb22(12)=abb22(12)*abb22(13)
      abb22(13)=spak2l3*abb22(28)*abb22(13)
      abb22(12)=abb22(12)+abb22(13)
      abb22(12)=spbl5l3*abb22(29)*abb22(23)*abb22(12)
      abb22(13)=2.0_ki*abb22(25)
      abb22(21)=-spbl5l3*abb22(13)*spak2l5*abb22(18)
      abb22(23)=abb22(26)*abb22(10)
      abb22(15)=abb22(23)+abb22(15)
      abb22(15)=-abb22(15)*abb22(19)
      abb22(19)=-NC*abb22(23)
      abb22(19)=-abb22(20)+abb22(19)
      abb22(19)=c2*abb22(19)
      abb22(20)=abb22(18)*abb22(22)
      abb22(15)=abb22(20)+abb22(19)+abb22(15)
      abb22(15)=abb22(15)*abb22(13)
      abb22(17)=-abb22(17)*abb22(18)
      abb22(18)=-spak2l3*abb22(27)*abb22(18)
      abb22(17)=abb22(18)+abb22(17)
      abb22(13)=abb22(17)*abb22(13)
      R2d22=abb22(14)
      rat2 = rat2 + R2d22
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='22' value='", &
          & R2d22, "'/>"
      end if
   end subroutine
end module p0_ubaru_httbar_abbrevd22h14_qp
