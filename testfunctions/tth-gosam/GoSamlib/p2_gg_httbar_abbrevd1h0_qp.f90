module     p2_gg_httbar_abbrevd1h0_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh0_qp
   implicit none
   private
   complex(ki), dimension(40), public :: abb1
   complex(ki), public :: R2d1
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_color_qp, only: TR
      use p2_gg_httbar_globalsl1_qp, only: epspow
      implicit none
      abb1(1)=1.0_ki/(-mT**2+es34)
      abb1(2)=sqrt(mT**2)
      abb1(3)=spbl5k2**(-1)
      abb1(4)=spbl4k2**(-1)
      abb1(5)=spak2l3**(-1)
      abb1(6)=spbl3k2**(-1)
      abb1(7)=abb1(2)*mT
      abb1(8)=i_*TR*e*gHT*abb1(1)*gs**4
      abb1(9)=abb1(7)*abb1(8)
      abb1(10)=1.0_ki/2.0_ki*c3
      abb1(11)=abb1(10)*abb1(4)
      abb1(12)=spbe2e1*abb1(11)*abb1(9)
      abb1(13)=abb1(8)*NC
      abb1(7)=abb1(13)*abb1(7)
      abb1(14)=abb1(7)*spbe2e1
      abb1(15)=1.0_ki/2.0_ki*abb1(4)
      abb1(16)=abb1(14)*abb1(15)
      abb1(17)=c2*abb1(16)
      abb1(18)=c1*abb1(4)
      abb1(19)=-abb1(18)*abb1(14)
      abb1(17)=-abb1(12)+abb1(17)+abb1(19)
      abb1(17)=spae2l3*spae1l5*abb1(17)
      abb1(19)=c2*abb1(4)
      abb1(14)=abb1(19)*abb1(14)
      abb1(16)=-c1*abb1(16)
      abb1(12)=abb1(12)+abb1(14)+abb1(16)
      abb1(12)=spae1l3*spae2l5*abb1(12)
      abb1(14)=c1+c2
      abb1(16)=spae1e2*spbe2e1
      abb1(20)=abb1(16)*abb1(3)
      abb1(21)=-abb1(20)*abb1(7)*abb1(14)
      abb1(22)=abb1(16)*c3
      abb1(23)=2.0_ki*abb1(22)
      abb1(24)=-abb1(3)*abb1(9)*abb1(23)
      abb1(21)=abb1(24)+abb1(21)
      abb1(21)=spal3l4*abb1(21)
      abb1(24)=abb1(19)+abb1(18)
      abb1(25)=abb1(24)*abb1(16)
      abb1(26)=-abb1(7)*abb1(25)
      abb1(27)=2.0_ki*abb1(9)
      abb1(28)=abb1(22)*abb1(4)
      abb1(27)=-abb1(28)*abb1(27)
      abb1(26)=abb1(27)+abb1(26)
      abb1(26)=spal3l5*abb1(26)
      abb1(12)=abb1(26)+abb1(12)+abb1(17)+abb1(21)
      abb1(12)=spbl3k2*abb1(12)
      abb1(17)=-abb1(19)+abb1(18)
      abb1(21)=mT**2
      abb1(26)=abb1(21)*abb1(8)
      abb1(27)=abb1(2)*NC
      abb1(29)=abb1(26)*abb1(27)
      abb1(30)=abb1(13)*mT**3
      abb1(29)=abb1(29)+abb1(30)
      abb1(30)=abb1(3)*spae1e2
      abb1(17)=abb1(30)*abb1(2)*abb1(29)*abb1(17)
      abb1(29)=abb1(7)*abb1(30)
      abb1(31)=abb1(29)*c2
      abb1(29)=abb1(29)*c1
      abb1(32)=-abb1(31)+abb1(29)
      abb1(33)=abb1(5)*mH**2*spak2l4*abb1(6)
      abb1(32)=abb1(32)*abb1(33)
      abb1(17)=abb1(17)+abb1(32)
      abb1(17)=3.0_ki/2.0_ki*abb1(17)
      abb1(17)=spbk2e2*abb1(17)
      abb1(30)=abb1(9)*abb1(10)*abb1(30)
      abb1(32)=abb1(30)-1.0_ki/2.0_ki*abb1(31)+abb1(29)
      abb1(34)=spbl3e2*spal3l4
      abb1(32)=abb1(32)*abb1(34)
      abb1(17)=abb1(32)+abb1(17)
      abb1(17)=spbk2e1*abb1(17)
      abb1(32)=mT+abb1(2)
      abb1(32)=abb1(8)*abb1(32)
      abb1(35)=spbe2e1*abb1(2)
      abb1(36)=abb1(32)*abb1(35)*abb1(10)
      abb1(37)=NC*mT
      abb1(27)=abb1(37)+abb1(27)
      abb1(27)=abb1(8)*abb1(27)
      abb1(35)=abb1(27)*abb1(35)
      abb1(37)=abb1(35)*c2
      abb1(35)=abb1(35)*c1
      abb1(38)=-abb1(36)+1.0_ki/2.0_ki*abb1(37)-abb1(35)
      abb1(38)=spae2l4*spae1l5*abb1(38)
      abb1(35)=abb1(36)+abb1(37)-1.0_ki/2.0_ki*abb1(35)
      abb1(35)=spae1l4*spae2l5*abb1(35)
      abb1(29)=-abb1(30)-abb1(31)+1.0_ki/2.0_ki*abb1(29)
      abb1(30)=spbl3e1*spal3l4
      abb1(29)=spbk2e2*abb1(29)*abb1(30)
      abb1(14)=-abb1(16)*abb1(27)*abb1(14)
      abb1(23)=-abb1(32)*abb1(23)
      abb1(14)=abb1(23)+abb1(14)
      abb1(14)=spal4l5*abb1(2)*abb1(14)
      abb1(12)=abb1(14)+abb1(29)+abb1(35)+abb1(38)+abb1(12)+abb1(17)
      abb1(9)=abb1(26)+abb1(9)
      abb1(14)=abb1(9)*abb1(11)
      abb1(17)=abb1(21)*abb1(13)
      abb1(7)=abb1(17)+abb1(7)
      abb1(21)=abb1(7)*c2
      abb1(23)=abb1(15)*abb1(21)
      abb1(27)=-abb1(7)*abb1(18)
      abb1(29)=1.0_ki/2.0_ki*abb1(8)
      abb1(29)=abb1(29)*c3
      abb1(31)=abb1(13)*c2
      abb1(13)=abb1(13)*c1
      abb1(32)=-1.0_ki/2.0_ki*abb1(31)+abb1(29)+abb1(13)
      abb1(35)=-abb1(32)*abb1(33)
      abb1(23)=abb1(35)-abb1(14)+abb1(23)+abb1(27)
      abb1(23)=spae1l5*abb1(23)
      abb1(27)=abb1(7)*c1
      abb1(35)=abb1(27)*abb1(3)
      abb1(36)=abb1(21)*abb1(3)
      abb1(37)=abb1(9)*abb1(3)
      abb1(10)=abb1(37)*abb1(10)
      abb1(38)=-abb1(10)-abb1(36)+1.0_ki/2.0_ki*abb1(35)
      abb1(38)=spae1l4*abb1(38)
      abb1(39)=c1*abb1(15)
      abb1(39)=-abb1(19)+abb1(39)
      abb1(40)=abb1(17)*abb1(3)
      abb1(39)=abb1(40)*abb1(39)
      abb1(26)=abb1(3)*abb1(26)
      abb1(11)=abb1(26)*abb1(11)
      abb1(39)=-abb1(11)+abb1(39)
      abb1(39)=spae1l3*spbl3k2*abb1(39)
      abb1(23)=abb1(39)+abb1(38)+abb1(23)
      abb1(23)=spbk2e2*abb1(23)
      abb1(32)=-spae1l5*abb1(32)*abb1(34)
      abb1(23)=abb1(32)+abb1(23)
      abb1(19)=-abb1(7)*abb1(19)
      abb1(32)=abb1(15)*abb1(27)
      abb1(29)=-1.0_ki/2.0_ki*abb1(13)+abb1(29)+abb1(31)
      abb1(34)=-abb1(29)*abb1(33)
      abb1(14)=abb1(34)-abb1(14)+abb1(19)+abb1(32)
      abb1(14)=spae2l5*abb1(14)
      abb1(10)=-abb1(10)+1.0_ki/2.0_ki*abb1(36)-abb1(35)
      abb1(10)=spae2l4*abb1(10)
      abb1(15)=c2*abb1(15)
      abb1(15)=abb1(15)-abb1(18)
      abb1(15)=abb1(40)*abb1(15)
      abb1(11)=-abb1(11)+abb1(15)
      abb1(11)=spae2l3*spbl3k2*abb1(11)
      abb1(10)=abb1(11)+abb1(10)+abb1(14)
      abb1(10)=spbk2e1*abb1(10)
      abb1(11)=-spae2l5*abb1(29)*abb1(30)
      abb1(10)=abb1(11)+abb1(10)
      abb1(11)=abb1(31)+abb1(13)
      abb1(11)=abb1(11)*abb1(16)
      abb1(8)=abb1(22)*abb1(8)
      abb1(8)=abb1(8)+1.0_ki/2.0_ki*abb1(11)
      abb1(11)=-spal3l4*abb1(8)
      abb1(7)=-abb1(7)*abb1(25)
      abb1(9)=-abb1(9)*abb1(28)
      abb1(8)=-abb1(8)*abb1(33)
      abb1(7)=abb1(8)+1.0_ki/2.0_ki*abb1(7)+abb1(9)
      abb1(8)=-abb1(21)-abb1(27)
      abb1(8)=abb1(20)*abb1(8)
      abb1(9)=-abb1(22)*abb1(37)
      abb1(8)=1.0_ki/2.0_ki*abb1(8)+abb1(9)
      abb1(9)=-abb1(20)*abb1(17)*abb1(24)
      abb1(13)=-abb1(28)*abb1(26)
      abb1(9)=1.0_ki/2.0_ki*abb1(9)+abb1(13)
      abb1(9)=spbl3k2*abb1(9)
      R2d1=0.0_ki
      rat2 = rat2 + R2d1
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='1' value='", &
          & R2d1, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd1h0_qp
