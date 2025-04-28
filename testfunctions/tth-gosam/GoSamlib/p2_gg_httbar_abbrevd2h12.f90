module     p2_gg_httbar_abbrevd2h12
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh12
   implicit none
   private
   complex(ki), dimension(40), public :: abb2
   complex(ki), public :: R2d2
   public :: init_abbrev
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
contains
   subroutine     init_abbrev()
      use p2_gg_httbar_config, only: deltaOS, &
     &    logfile, debug_nlo_diagrams
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_model
      use p2_gg_httbar_color, only: TR
      use p2_gg_httbar_globalsl1, only: epspow
      implicit none
      abb2(1)=1.0_ki/(mH**2+mT**2-es34-es45+es12)
      abb2(2)=sqrt(mT**2)
      abb2(3)=spak2l4**(-1)
      abb2(4)=spak2l5**(-1)
      abb2(5)=spak2l3**(-1)
      abb2(6)=spbl3k2**(-1)
      abb2(7)=abb2(2)*mT
      abb2(8)=i_*TR*e*gHT*abb2(1)*gs**4
      abb2(9)=abb2(7)*abb2(8)
      abb2(10)=1.0_ki/2.0_ki*c3
      abb2(11)=abb2(10)*abb2(4)
      abb2(12)=spae1e2*abb2(11)*abb2(9)
      abb2(13)=abb2(8)*NC
      abb2(7)=abb2(13)*abb2(7)
      abb2(14)=abb2(7)*spae1e2
      abb2(15)=c2*abb2(4)
      abb2(16)=-abb2(15)*abb2(14)
      abb2(17)=1.0_ki/2.0_ki*abb2(4)
      abb2(18)=abb2(14)*abb2(17)
      abb2(19)=c1*abb2(18)
      abb2(16)=-abb2(12)+abb2(16)+abb2(19)
      abb2(16)=spbl3e2*spbl4e1*abb2(16)
      abb2(18)=-c2*abb2(18)
      abb2(19)=c1*abb2(4)
      abb2(14)=abb2(19)*abb2(14)
      abb2(12)=abb2(12)+abb2(18)+abb2(14)
      abb2(12)=spbl3e1*spbl4e2*abb2(12)
      abb2(14)=c1+c2
      abb2(18)=spbe2e1*spae1e2
      abb2(20)=abb2(18)*abb2(3)
      abb2(21)=-abb2(20)*abb2(7)*abb2(14)
      abb2(22)=abb2(18)*c3
      abb2(23)=2.0_ki*abb2(22)
      abb2(24)=-abb2(3)*abb2(9)*abb2(23)
      abb2(21)=abb2(24)+abb2(21)
      abb2(21)=spbl5l3*abb2(21)
      abb2(24)=abb2(15)+abb2(19)
      abb2(25)=abb2(24)*abb2(18)
      abb2(26)=-abb2(7)*abb2(25)
      abb2(27)=2.0_ki*abb2(9)
      abb2(28)=abb2(22)*abb2(4)
      abb2(27)=-abb2(28)*abb2(27)
      abb2(26)=abb2(27)+abb2(26)
      abb2(26)=spbl4l3*abb2(26)
      abb2(12)=abb2(26)+abb2(12)+abb2(16)+abb2(21)
      abb2(12)=spak2l3*abb2(12)
      abb2(16)=abb2(15)-abb2(19)
      abb2(21)=mT**2
      abb2(26)=abb2(21)*abb2(8)
      abb2(27)=abb2(2)*NC
      abb2(29)=abb2(26)*abb2(27)
      abb2(30)=abb2(13)*mT**3
      abb2(29)=abb2(29)+abb2(30)
      abb2(30)=abb2(3)*spbe2e1
      abb2(16)=abb2(30)*abb2(2)*abb2(29)*abb2(16)
      abb2(29)=abb2(7)*abb2(30)
      abb2(31)=abb2(29)*c2
      abb2(29)=abb2(29)*c1
      abb2(32)=abb2(31)-abb2(29)
      abb2(33)=abb2(5)*mH**2*spbl5k2*abb2(6)
      abb2(32)=abb2(32)*abb2(33)
      abb2(16)=abb2(16)+abb2(32)
      abb2(16)=3.0_ki/2.0_ki*abb2(16)
      abb2(16)=spae2k2*abb2(16)
      abb2(30)=abb2(9)*abb2(10)*abb2(30)
      abb2(32)=abb2(30)+abb2(31)-1.0_ki/2.0_ki*abb2(29)
      abb2(34)=spae2l3*spbl5l3
      abb2(32)=abb2(32)*abb2(34)
      abb2(16)=abb2(32)+abb2(16)
      abb2(16)=spae1k2*abb2(16)
      abb2(32)=mT+abb2(2)
      abb2(32)=abb2(8)*abb2(32)
      abb2(35)=spae1e2*abb2(2)
      abb2(36)=abb2(32)*abb2(35)*abb2(10)
      abb2(37)=NC*mT
      abb2(27)=abb2(37)+abb2(27)
      abb2(27)=abb2(8)*abb2(27)
      abb2(35)=abb2(27)*abb2(35)
      abb2(37)=abb2(35)*c2
      abb2(35)=abb2(35)*c1
      abb2(38)=-abb2(36)-abb2(37)+1.0_ki/2.0_ki*abb2(35)
      abb2(38)=spbl5e2*spbl4e1*abb2(38)
      abb2(35)=abb2(36)-1.0_ki/2.0_ki*abb2(37)+abb2(35)
      abb2(35)=spbl5e1*spbl4e2*abb2(35)
      abb2(29)=-abb2(30)+1.0_ki/2.0_ki*abb2(31)-abb2(29)
      abb2(30)=spae1l3*spbl5l3
      abb2(29)=spae2k2*abb2(29)*abb2(30)
      abb2(14)=abb2(18)*abb2(27)*abb2(14)
      abb2(23)=abb2(32)*abb2(23)
      abb2(14)=abb2(23)+abb2(14)
      abb2(14)=spbl5l4*abb2(2)*abb2(14)
      abb2(12)=abb2(14)+abb2(29)+abb2(35)+abb2(38)+abb2(12)+abb2(16)
      abb2(9)=abb2(26)+abb2(9)
      abb2(14)=abb2(9)*abb2(11)
      abb2(16)=abb2(21)*abb2(13)
      abb2(7)=abb2(16)+abb2(7)
      abb2(21)=abb2(7)*c2
      abb2(23)=abb2(17)*abb2(21)
      abb2(27)=-abb2(7)*abb2(19)
      abb2(29)=1.0_ki/2.0_ki*abb2(8)
      abb2(29)=abb2(29)*c3
      abb2(31)=abb2(13)*c2
      abb2(13)=abb2(13)*c1
      abb2(32)=-1.0_ki/2.0_ki*abb2(31)+abb2(29)+abb2(13)
      abb2(35)=-abb2(32)*abb2(33)
      abb2(23)=abb2(35)-abb2(14)+abb2(23)+abb2(27)
      abb2(23)=spbl4e2*abb2(23)
      abb2(27)=abb2(7)*c1
      abb2(35)=abb2(27)*abb2(3)
      abb2(36)=abb2(21)*abb2(3)
      abb2(37)=abb2(9)*abb2(3)
      abb2(10)=abb2(37)*abb2(10)
      abb2(38)=-abb2(10)-abb2(36)+1.0_ki/2.0_ki*abb2(35)
      abb2(38)=spbl5e2*abb2(38)
      abb2(39)=c1*abb2(17)
      abb2(39)=-abb2(15)+abb2(39)
      abb2(40)=abb2(16)*abb2(3)
      abb2(39)=abb2(40)*abb2(39)
      abb2(26)=abb2(3)*abb2(26)
      abb2(11)=abb2(26)*abb2(11)
      abb2(39)=-abb2(11)+abb2(39)
      abb2(39)=spbl3e2*spak2l3*abb2(39)
      abb2(23)=abb2(39)+abb2(38)+abb2(23)
      abb2(23)=spae1k2*abb2(23)
      abb2(30)=-spbl4e2*abb2(32)*abb2(30)
      abb2(23)=abb2(30)+abb2(23)
      abb2(15)=-abb2(7)*abb2(15)
      abb2(30)=abb2(17)*abb2(27)
      abb2(29)=-1.0_ki/2.0_ki*abb2(13)+abb2(29)+abb2(31)
      abb2(32)=-abb2(29)*abb2(33)
      abb2(14)=abb2(32)-abb2(14)+abb2(15)+abb2(30)
      abb2(14)=spbl4e1*abb2(14)
      abb2(10)=-abb2(10)+1.0_ki/2.0_ki*abb2(36)-abb2(35)
      abb2(10)=spbl5e1*abb2(10)
      abb2(15)=c2*abb2(17)
      abb2(15)=abb2(15)-abb2(19)
      abb2(15)=abb2(40)*abb2(15)
      abb2(11)=-abb2(11)+abb2(15)
      abb2(11)=spbl3e1*spak2l3*abb2(11)
      abb2(10)=abb2(11)+abb2(10)+abb2(14)
      abb2(10)=spae2k2*abb2(10)
      abb2(11)=-spbl4e1*abb2(29)*abb2(34)
      abb2(10)=abb2(11)+abb2(10)
      abb2(11)=abb2(31)+abb2(13)
      abb2(11)=abb2(11)*abb2(18)
      abb2(8)=abb2(22)*abb2(8)
      abb2(8)=abb2(8)+1.0_ki/2.0_ki*abb2(11)
      abb2(11)=-spbl5l3*abb2(8)
      abb2(13)=-abb2(21)-abb2(27)
      abb2(13)=abb2(20)*abb2(13)
      abb2(14)=-abb2(22)*abb2(37)
      abb2(13)=1.0_ki/2.0_ki*abb2(13)+abb2(14)
      abb2(7)=-abb2(7)*abb2(25)
      abb2(9)=-abb2(9)*abb2(28)
      abb2(8)=-abb2(8)*abb2(33)
      abb2(7)=abb2(8)+1.0_ki/2.0_ki*abb2(7)+abb2(9)
      abb2(8)=-abb2(20)*abb2(16)*abb2(24)
      abb2(9)=-abb2(28)*abb2(26)
      abb2(8)=1.0_ki/2.0_ki*abb2(8)+abb2(9)
      abb2(8)=spak2l3*abb2(8)
      R2d2=0.0_ki
      rat2 = rat2 + R2d2
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='2' value='", &
          & R2d2, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd2h12
