module     p2_gg_httbar_abbrevd101h12
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_kinematics, only: epstensor
   use p2_gg_httbar_globalsh12
   implicit none
   private
   complex(ki), dimension(49), public :: abb101
   complex(ki), public :: R2d101
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
      abb101(1)=sqrt(mT**2)
      abb101(2)=es45**(-1)
      abb101(3)=spak2l4**(-1)
      abb101(4)=spak2l5**(-1)
      abb101(5)=spak2l3**(-1)
      abb101(6)=spbl3k2**(-1)
      abb101(7)=c1-c2
      abb101(8)=gs**4*i_*TR*mT*e*gHT*abb101(2)
      abb101(9)=abb101(8)*abb101(7)*abb101(1)**3
      abb101(10)=-abb101(3)*abb101(9)
      abb101(11)=spbe2e1*spae2k2
      abb101(12)=abb101(10)*abb101(11)
      abb101(13)=spae1k2*abb101(12)*spbl5k2
      abb101(9)=-abb101(4)*abb101(9)
      abb101(14)=abb101(9)*spae1k2
      abb101(15)=abb101(11)*spbl4k2
      abb101(16)=abb101(14)*abb101(15)
      abb101(13)=abb101(13)+abb101(16)
      abb101(16)=mH**2*abb101(6)*abb101(5)
      abb101(17)=-abb101(16)+1.0_ki
      abb101(13)=abb101(13)*abb101(17)
      abb101(17)=2.0_ki*abb101(12)
      abb101(18)=spbl5k1*abb101(17)
      abb101(19)=abb101(9)*abb101(11)
      abb101(20)=2.0_ki*abb101(19)
      abb101(21)=spbl4k1*abb101(20)
      abb101(18)=abb101(21)+abb101(18)
      abb101(18)=spae1k1*abb101(18)
      abb101(21)=abb101(10)*spbl5e2
      abb101(22)=abb101(9)*spbl4e2
      abb101(21)=abb101(21)+abb101(22)
      abb101(22)=2.0_ki*abb101(21)
      abb101(23)=spak1k2*spbk1e1
      abb101(24)=abb101(23)*spae1e2
      abb101(22)=abb101(24)*abb101(22)
      abb101(25)=spae1e2*abb101(21)
      abb101(26)=spbl3e1*spak2l3
      abb101(27)=abb101(25)*abb101(26)
      abb101(7)=abb101(7)*abb101(8)*abb101(1)
      abb101(8)=-abb101(3)*abb101(7)
      abb101(28)=abb101(8)*spbl5k2
      abb101(11)=abb101(28)*abb101(11)
      abb101(29)=abb101(11)*spae1k1
      abb101(30)=-spak2l3*abb101(29)
      abb101(7)=-abb101(4)*abb101(7)
      abb101(31)=abb101(7)*spak2l3
      abb101(32)=abb101(31)*spae1k1
      abb101(33)=-abb101(15)*abb101(32)
      abb101(30)=abb101(30)+abb101(33)
      abb101(30)=spbl3k1*abb101(30)
      abb101(12)=-spbl5l3*abb101(12)
      abb101(19)=-spbl4l3*abb101(19)
      abb101(12)=abb101(12)+abb101(19)
      abb101(12)=spae1l3*abb101(12)
      abb101(12)=abb101(12)+abb101(30)+abb101(27)+abb101(22)+abb101(18)+abb101(&
      &13)
      abb101(13)=spae1k2*abb101(11)
      abb101(18)=abb101(7)*spae1k2
      abb101(19)=abb101(15)*abb101(18)
      abb101(13)=abb101(13)+abb101(19)
      abb101(19)=-spae1k2*abb101(21)
      abb101(21)=abb101(8)*spak2l3
      abb101(22)=abb101(21)*spbl5e2
      abb101(27)=spae1k1*abb101(22)
      abb101(30)=spbl4e2*abb101(32)
      abb101(27)=abb101(27)+abb101(30)
      abb101(27)=spbl3k1*abb101(27)
      abb101(19)=abb101(27)+abb101(19)
      abb101(27)=abb101(8)*spbl5e2
      abb101(30)=-spae1k2*abb101(27)
      abb101(33)=-spbl4e2*abb101(18)
      abb101(30)=abb101(30)+abb101(33)
      abb101(33)=abb101(7)*spbl4k2
      abb101(28)=abb101(33)+abb101(28)
      abb101(33)=spae2k2*abb101(28)
      abb101(34)=abb101(16)-2.0_ki
      abb101(34)=abb101(33)*abb101(34)
      abb101(35)=-abb101(23)*abb101(34)
      abb101(9)=abb101(9)*spbl4e1
      abb101(36)=abb101(10)*spbl5e1
      abb101(9)=abb101(9)+abb101(36)
      abb101(36)=-spae2k2*abb101(9)
      abb101(26)=abb101(33)*abb101(26)
      abb101(33)=abb101(8)*spbl5l3
      abb101(37)=abb101(7)*spbl4l3
      abb101(33)=abb101(33)+abb101(37)
      abb101(37)=abb101(33)*spak1l3
      abb101(38)=spbk1e1*spae2k2
      abb101(39)=-abb101(38)*abb101(37)
      abb101(26)=abb101(39)+abb101(26)+abb101(36)+abb101(35)
      abb101(35)=abb101(8)*spae2k2
      abb101(36)=-spbl5e1*abb101(35)
      abb101(39)=abb101(7)*spae2k2
      abb101(40)=-spbl4e1*abb101(39)
      abb101(36)=abb101(36)+abb101(40)
      abb101(40)=spae1k2*spbe2e1
      abb101(10)=-abb101(10)*abb101(40)
      abb101(41)=abb101(21)*spbe2e1
      abb101(42)=spbl3k1*spae1k1*abb101(41)
      abb101(10)=abb101(10)+abb101(42)
      abb101(40)=-abb101(8)*abb101(40)
      abb101(23)=2.0_ki*abb101(23)
      abb101(42)=-abb101(8)*abb101(23)
      abb101(21)=-spbl3e1*abb101(21)
      abb101(21)=abb101(42)+abb101(21)
      abb101(32)=spbl3k1*abb101(32)
      abb101(14)=-abb101(14)+abb101(32)
      abb101(14)=spbe2e1*abb101(14)
      abb101(18)=-spbe2e1*abb101(18)
      abb101(23)=-abb101(7)*abb101(23)
      abb101(32)=-spbl3e1*abb101(31)
      abb101(23)=abb101(23)+abb101(32)
      abb101(32)=abb101(11)*spak2l3
      abb101(42)=-abb101(15)*abb101(31)
      abb101(32)=-abb101(32)+abb101(42)
      abb101(42)=spbl4e2*abb101(31)
      abb101(22)=abb101(22)+abb101(42)
      abb101(31)=spbe2e1*abb101(31)
      abb101(42)=-spbl5l3*abb101(35)
      abb101(43)=-spbl4l3*abb101(39)
      abb101(42)=abb101(42)+abb101(43)
      abb101(9)=-spae1e2*abb101(9)
      abb101(16)=abb101(28)*abb101(16)
      abb101(24)=-abb101(24)*abb101(16)
      abb101(28)=spbk1e1*spae1e2
      abb101(37)=-abb101(28)*abb101(37)
      abb101(9)=abb101(37)+abb101(24)+abb101(9)
      abb101(24)=-spbl5e1*abb101(8)
      abb101(37)=-spbl4e1*abb101(7)
      abb101(24)=abb101(24)+abb101(37)
      abb101(24)=spae1e2*abb101(24)
      abb101(37)=abb101(8)*spae1k1
      abb101(43)=spbl5k1*abb101(37)
      abb101(44)=abb101(7)*spae1k1
      abb101(45)=spbl4k1*abb101(44)
      abb101(43)=abb101(43)+abb101(45)
      abb101(45)=spae1k2*abb101(16)
      abb101(46)=spae1l3*abb101(33)
      abb101(43)=abb101(45)+abb101(46)-2.0_ki*abb101(43)
      abb101(45)=2.0_ki*abb101(8)
      abb101(46)=2.0_ki*abb101(7)
      abb101(33)=-spae1e2*abb101(33)
      abb101(25)=2.0_ki*abb101(25)
      abb101(16)=-spae1e2*abb101(16)
      abb101(11)=-spak1k2*abb101(11)
      abb101(47)=abb101(7)*spak1k2
      abb101(48)=-abb101(15)*abb101(47)
      abb101(11)=abb101(11)+abb101(48)
      abb101(27)=spak1k2*abb101(27)
      abb101(48)=spbl4e2*abb101(47)
      abb101(27)=abb101(27)+abb101(48)
      abb101(48)=spbe2e1*abb101(8)*spak1k2
      abb101(47)=spbe2e1*abb101(47)
      abb101(35)=spbl5k1*abb101(35)
      abb101(39)=spbl4k1*abb101(39)
      abb101(35)=abb101(35)+abb101(39)
      abb101(39)=spbl5k1*abb101(8)
      abb101(49)=spbl4k1*abb101(7)
      abb101(39)=abb101(39)+abb101(49)
      abb101(39)=spae1e2*abb101(39)
      abb101(15)=-abb101(15)*abb101(44)
      abb101(15)=-abb101(29)+abb101(15)
      abb101(29)=spbl5e2*abb101(37)
      abb101(49)=spbl4e2*abb101(44)
      abb101(29)=abb101(29)+abb101(49)
      abb101(37)=spbe2e1*abb101(37)
      abb101(44)=spbe2e1*abb101(44)
      abb101(49)=abb101(8)*abb101(38)
      abb101(8)=abb101(8)*abb101(28)
      abb101(38)=abb101(7)*abb101(38)
      abb101(7)=abb101(7)*abb101(28)
      R2d101=0.0_ki
      rat2 = rat2 + R2d101
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='101' value='", &
          & R2d101, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd101h12
