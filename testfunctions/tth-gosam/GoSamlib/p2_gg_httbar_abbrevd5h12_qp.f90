module     p2_gg_httbar_abbrevd5h12_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh12_qp
   implicit none
   private
   complex(ki), dimension(35), public :: abb5
   complex(ki), public :: R2d5
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
      abb5(1)=sqrt(mT**2)
      abb5(2)=spak2l4**(-1)
      abb5(3)=spak2l5**(-1)
      abb5(4)=spak2l3**(-1)
      abb5(5)=spbl3k2**(-1)
      abb5(6)=i_*TR*e*gHT
      abb5(7)=abb5(6)*c3
      abb5(8)=mT*abb5(1)
      abb5(9)=abb5(8)*abb5(7)
      abb5(10)=gs**4
      abb5(11)=1.0_ki/2.0_ki*abb5(10)
      abb5(11)=abb5(11)*abb5(9)
      abb5(6)=abb5(6)*abb5(10)*NC
      abb5(12)=abb5(6)*c1
      abb5(13)=abb5(12)*abb5(8)
      abb5(6)=abb5(6)*c2
      abb5(14)=abb5(6)*abb5(8)
      abb5(15)=-abb5(11)-abb5(14)+1.0_ki/2.0_ki*abb5(13)
      abb5(16)=abb5(2)*spbl5e2*spae1k2
      abb5(17)=-abb5(15)*abb5(16)
      abb5(11)=-abb5(11)-abb5(13)+1.0_ki/2.0_ki*abb5(14)
      abb5(18)=abb5(3)*spbl4e2*spae1k2
      abb5(19)=-abb5(11)*abb5(18)
      abb5(17)=abb5(17)+abb5(19)
      abb5(17)=spbl3e1*spae2l3*abb5(17)
      abb5(19)=abb5(2)*spbl5e1*spae2k2
      abb5(11)=-abb5(11)*abb5(19)
      abb5(20)=abb5(3)*spbl4e1*spae2k2
      abb5(15)=-abb5(15)*abb5(20)
      abb5(11)=abb5(11)+abb5(15)
      abb5(11)=spae1l3*spbl3e2*abb5(11)
      abb5(15)=abb5(6)-abb5(12)
      abb5(21)=abb5(3)*abb5(2)*spae1k2*spae2k2
      abb5(22)=3.0_ki/2.0_ki*abb5(21)
      abb5(15)=abb5(22)*abb5(15)*spbe2e1
      abb5(8)=abb5(8)**2*abb5(15)
      abb5(22)=-abb5(12)+1.0_ki/2.0_ki*abb5(6)
      abb5(23)=abb5(1)**2
      abb5(24)=abb5(23)*spae1e2
      abb5(25)=-abb5(24)*abb5(22)
      abb5(26)=1.0_ki/2.0_ki*abb5(7)
      abb5(27)=abb5(10)*spae1e2
      abb5(28)=abb5(26)*abb5(27)
      abb5(29)=abb5(28)*abb5(23)
      abb5(25)=abb5(29)+abb5(25)
      abb5(30)=spbl5e1*spbl4e2
      abb5(25)=abb5(25)*abb5(30)
      abb5(31)=-abb5(6)+1.0_ki/2.0_ki*abb5(12)
      abb5(24)=abb5(24)*abb5(31)
      abb5(24)=-abb5(29)+abb5(24)
      abb5(29)=spbl4e1*spbl5e2
      abb5(24)=abb5(24)*abb5(29)
      abb5(6)=abb5(12)+abb5(6)
      abb5(12)=spbe2e1*spae1e2
      abb5(6)=abb5(12)*abb5(6)
      abb5(27)=abb5(27)*spbe2e1
      abb5(7)=abb5(7)*abb5(27)
      abb5(32)=abb5(6)+2.0_ki*abb5(7)
      abb5(32)=abb5(32)*spbl5l4
      abb5(23)=abb5(23)*abb5(32)
      abb5(33)=abb5(14)+abb5(13)
      abb5(12)=abb5(12)*abb5(33)
      abb5(27)=abb5(9)*abb5(27)
      abb5(33)=abb5(27)+1.0_ki/2.0_ki*abb5(12)
      abb5(34)=-spbl5l3*abb5(2)
      abb5(35)=-spbl4l3*abb5(3)
      abb5(34)=abb5(34)+abb5(35)
      abb5(33)=spak2l3*abb5(33)*abb5(34)
      abb5(8)=abb5(33)+abb5(23)+abb5(11)+abb5(17)+abb5(24)+abb5(8)+abb5(25)
      abb5(11)=mT**2
      abb5(15)=abb5(11)*abb5(15)
      abb5(17)=-spae1e2*abb5(22)
      abb5(17)=abb5(28)+abb5(17)
      abb5(17)=abb5(17)*abb5(30)
      abb5(23)=spae1e2*abb5(31)
      abb5(23)=-abb5(28)+abb5(23)
      abb5(23)=abb5(23)*abb5(29)
      abb5(15)=abb5(32)+abb5(23)+abb5(15)+abb5(17)
      abb5(9)=abb5(9)*abb5(10)
      abb5(17)=abb5(9)-abb5(13)+2.0_ki*abb5(14)
      abb5(16)=abb5(17)*abb5(16)
      abb5(9)=abb5(9)-abb5(14)+2.0_ki*abb5(13)
      abb5(13)=abb5(9)*abb5(18)
      abb5(13)=abb5(16)+abb5(13)
      abb5(9)=abb5(9)*abb5(19)
      abb5(14)=abb5(17)*abb5(20)
      abb5(9)=abb5(9)+abb5(14)
      abb5(10)=abb5(26)*abb5(10)
      abb5(14)=abb5(22)-abb5(10)
      abb5(16)=abb5(14)*abb5(30)
      abb5(17)=-spae1l3*abb5(16)
      abb5(11)=abb5(21)*abb5(11)
      abb5(14)=spbl3e1*abb5(14)*abb5(11)
      abb5(10)=abb5(31)-abb5(10)
      abb5(18)=abb5(10)*abb5(29)
      abb5(19)=-spae2l3*abb5(18)
      abb5(10)=spbl3e2*abb5(10)*abb5(11)
      abb5(11)=abb5(4)*abb5(5)*mH**2
      abb5(16)=-spae1k2*abb5(11)*abb5(16)
      abb5(18)=-spae2k2*abb5(11)*abb5(18)
      abb5(6)=abb5(7)+1.0_ki/2.0_ki*abb5(6)
      abb5(7)=-spbl4l3*abb5(6)
      abb5(20)=spbl5l3*abb5(6)
      abb5(12)=abb5(12)+2.0_ki*abb5(27)
      abb5(21)=abb5(2)*abb5(12)
      abb5(6)=abb5(6)*abb5(11)
      abb5(11)=-spbl4k2*abb5(6)
      abb5(11)=abb5(11)+abb5(21)
      abb5(12)=abb5(3)*abb5(12)
      abb5(6)=spbl5k2*abb5(6)
      abb5(6)=abb5(6)+abb5(12)
      R2d5=0.0_ki
      rat2 = rat2 + R2d5
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='5' value='", &
          & R2d5, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd5h12_qp
