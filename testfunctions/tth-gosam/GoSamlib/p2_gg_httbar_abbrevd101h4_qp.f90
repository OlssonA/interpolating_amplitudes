module     p2_gg_httbar_abbrevd101h4_qp
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_kinematics_qp, only: epstensor
   use p2_gg_httbar_globalsh4_qp
   implicit none
   private
   complex(ki), dimension(62), public :: abb101
   complex(ki), public :: R2d101
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
      abb101(1)=sqrt(mT**2)
      abb101(2)=es45**(-1)
      abb101(3)=spak2l3**(-1)
      abb101(4)=spbl3k2**(-1)
      abb101(5)=spak2l4**(-1)
      abb101(6)=spbl5k2**(-1)
      abb101(7)=c1-c2
      abb101(7)=abb101(7)*i_*TR*e*gHT*abb101(2)*gs**4
      abb101(8)=-abb101(1)**3*abb101(7)
      abb101(9)=abb101(8)*spae1e2
      abb101(10)=abb101(9)*spbl4e2
      abb101(11)=abb101(3)*abb101(4)*mH**2
      abb101(12)=-spbk2e1*abb101(10)*abb101(11)
      abb101(13)=abb101(8)*spbl4e1
      abb101(14)=spbk2e2*spae1e2
      abb101(15)=abb101(14)*abb101(13)
      abb101(12)=abb101(15)+abb101(12)
      abb101(12)=spak2l5*abb101(12)
      abb101(15)=abb101(8)*spbe2e1
      abb101(16)=abb101(15)*spae2l5
      abb101(17)=-spae1k2*abb101(11)*abb101(16)
      abb101(18)=abb101(8)*spae1l5
      abb101(19)=spae2k2*spbe2e1
      abb101(20)=abb101(19)*abb101(18)
      abb101(17)=abb101(20)+abb101(17)
      abb101(17)=spbl4k2*abb101(17)
      abb101(7)=abb101(1)*abb101(7)
      abb101(20)=abb101(7)*spbl4k2
      abb101(21)=abb101(11)*abb101(20)
      abb101(22)=abb101(21)*spak2l5
      abb101(23)=mT**2*abb101(6)*abb101(5)
      abb101(8)=abb101(8)*abb101(23)
      abb101(24)=abb101(22)+2.0_ki*abb101(8)
      abb101(25)=abb101(14)*abb101(24)
      abb101(26)=-spak1k2*abb101(25)
      abb101(27)=abb101(7)*spak2l5
      abb101(28)=abb101(14)*abb101(27)
      abb101(29)=abb101(28)*spbl4l3
      abb101(30)=-spak1l3*abb101(29)
      abb101(31)=2.0_ki*abb101(10)
      abb101(32)=spak1l5*abb101(31)
      abb101(26)=abb101(30)+abb101(32)+abb101(26)
      abb101(26)=spbk1e1*abb101(26)
      abb101(24)=abb101(19)*abb101(24)
      abb101(30)=-spbk2k1*abb101(24)
      abb101(32)=2.0_ki*abb101(16)
      abb101(33)=spbl4k1*abb101(32)
      abb101(20)=abb101(20)*spae2k2
      abb101(34)=abb101(20)*spbe2e1
      abb101(35)=abb101(34)*spal3l5
      abb101(36)=-spbl3k1*abb101(35)
      abb101(30)=abb101(36)+abb101(33)+abb101(30)
      abb101(30)=spae1k1*abb101(30)
      abb101(14)=-spak2l3*abb101(8)*abb101(14)
      abb101(10)=-spal3l5*abb101(10)
      abb101(10)=abb101(10)+abb101(14)
      abb101(10)=spbl3e1*abb101(10)
      abb101(14)=-spbl3k2*abb101(8)*abb101(19)
      abb101(16)=-spbl4l3*abb101(16)
      abb101(14)=abb101(16)+abb101(14)
      abb101(14)=spae1l3*abb101(14)
      abb101(10)=abb101(14)+abb101(10)+abb101(30)+abb101(26)+abb101(17)+abb101(&
      &12)
      abb101(12)=-spbl4e1*abb101(28)
      abb101(14)=-spae1l5*abb101(34)
      abb101(12)=abb101(14)+abb101(12)
      abb101(14)=abb101(22)+abb101(8)
      abb101(16)=spae1k2*spbk2e2*abb101(14)
      abb101(11)=abb101(11)*abb101(27)
      abb101(17)=spae1k1*abb101(11)*spbk2k1
      abb101(18)=-abb101(18)+abb101(17)
      abb101(18)=spbl4e2*abb101(18)
      abb101(19)=abb101(27)*spbk2e2
      abb101(22)=2.0_ki*abb101(19)
      abb101(26)=spbl4k1*spae1k1
      abb101(27)=-abb101(26)*abb101(22)
      abb101(30)=spae1l3*spbl4l3
      abb101(19)=abb101(19)*abb101(30)
      abb101(23)=abb101(23)*abb101(7)
      abb101(33)=abb101(23)*spak2l3
      abb101(36)=abb101(33)*spae1k1
      abb101(37)=spbk2e2*abb101(36)
      abb101(38)=abb101(7)*spbl4e2
      abb101(39)=abb101(38)*spal3l5
      abb101(40)=spae1k1*abb101(39)
      abb101(37)=abb101(40)+abb101(37)
      abb101(37)=spbl3k1*abb101(37)
      abb101(16)=abb101(37)+abb101(19)+abb101(27)+abb101(16)+abb101(18)
      abb101(18)=abb101(23)*spbk2e2
      abb101(19)=-spae1k2*abb101(18)
      abb101(27)=spae1l5*abb101(38)
      abb101(19)=abb101(19)+abb101(27)
      abb101(14)=spbk2e1*spae2k2*abb101(14)
      abb101(27)=spbk1e1*abb101(21)*spak1k2
      abb101(13)=-abb101(13)+abb101(27)
      abb101(13)=spae2l5*abb101(13)
      abb101(37)=2.0_ki*abb101(20)
      abb101(40)=spak1l5*spbk1e1
      abb101(41)=-abb101(40)*abb101(37)
      abb101(42)=spbl3e1*spal3l5
      abb101(20)=abb101(20)*abb101(42)
      abb101(43)=abb101(23)*spbl3k2
      abb101(44)=abb101(43)*spbk1e1
      abb101(45)=spae2k2*abb101(44)
      abb101(46)=abb101(7)*spae2l5
      abb101(47)=abb101(46)*spbl4l3
      abb101(48)=spbk1e1*abb101(47)
      abb101(45)=abb101(48)+abb101(45)
      abb101(45)=spak1l3*abb101(45)
      abb101(13)=abb101(45)+abb101(20)+abb101(41)+abb101(14)+abb101(13)
      abb101(14)=abb101(23)*spae2k2
      abb101(20)=-spbk2e1*abb101(14)
      abb101(41)=spbl4e1*abb101(46)
      abb101(20)=abb101(20)+abb101(41)
      abb101(27)=spae1e2*abb101(27)
      abb101(9)=-spbl4e1*abb101(9)
      abb101(41)=abb101(7)*spae1e2
      abb101(45)=abb101(41)*spbl4l3
      abb101(48)=abb101(45)*spak1l3*spbk1e1
      abb101(9)=abb101(48)+abb101(27)+abb101(9)
      abb101(27)=spbl4e1*abb101(41)
      abb101(48)=-spae1k2*abb101(21)
      abb101(49)=2.0_ki*abb101(7)
      abb101(26)=abb101(49)*abb101(26)
      abb101(30)=-abb101(7)*abb101(30)
      abb101(26)=abb101(30)+abb101(48)+abb101(26)
      abb101(17)=spbe2e1*abb101(17)
      abb101(15)=-spae1l5*abb101(15)
      abb101(30)=abb101(7)*spbe2e1
      abb101(48)=abb101(30)*spal3l5
      abb101(50)=abb101(48)*spbl3k1*spae1k1
      abb101(15)=abb101(50)+abb101(17)+abb101(15)
      abb101(17)=spae1l5*abb101(30)
      abb101(50)=-spbk2e1*abb101(11)
      abb101(40)=abb101(49)*abb101(40)
      abb101(7)=-abb101(7)*abb101(42)
      abb101(7)=abb101(7)+abb101(50)+abb101(40)
      abb101(40)=abb101(33)*spbk2e2
      abb101(39)=abb101(39)+abb101(40)
      abb101(40)=abb101(43)*spae2k2
      abb101(40)=abb101(47)+abb101(40)
      abb101(42)=abb101(8)*spae1k2
      abb101(36)=spbl3k1*abb101(36)
      abb101(36)=abb101(42)+abb101(36)
      abb101(36)=spbe2e1*abb101(36)
      abb101(42)=abb101(23)*spbe2e1
      abb101(47)=-spae1k2*abb101(42)
      abb101(50)=abb101(23)*spbk1e1
      abb101(51)=spak1k2*abb101(50)
      abb101(52)=-spbl3e1*abb101(33)
      abb101(51)=-2.0_ki*abb101(51)+abb101(52)
      abb101(33)=abb101(33)*spbe2e1
      abb101(8)=abb101(8)*spbk2e1
      abb101(44)=spak1l3*abb101(44)
      abb101(8)=abb101(8)+abb101(44)
      abb101(8)=spae1e2*abb101(8)
      abb101(44)=abb101(23)*spae1e2
      abb101(52)=-spbk2e1*abb101(44)
      abb101(53)=abb101(23)*spae1k1
      abb101(54)=spbk2k1*abb101(53)
      abb101(55)=-spae1l3*abb101(43)
      abb101(54)=-2.0_ki*abb101(54)+abb101(55)
      abb101(43)=abb101(43)*spae1e2
      abb101(55)=abb101(11)*spbl4e2
      abb101(11)=abb101(11)*spbe2e1
      abb101(23)=2.0_ki*abb101(23)
      abb101(56)=abb101(21)*spae2l5
      abb101(21)=abb101(21)*spae1e2
      abb101(57)=spak1l5*abb101(34)
      abb101(18)=spak1k2*abb101(18)
      abb101(58)=-spak1l5*abb101(38)
      abb101(18)=abb101(18)+abb101(58)
      abb101(58)=-spak1l5*abb101(30)
      abb101(42)=spak1k2*abb101(42)
      abb101(59)=spbl4k1*abb101(28)
      abb101(14)=spbk2k1*abb101(14)
      abb101(60)=-spbl4k1*abb101(46)
      abb101(14)=abb101(14)+abb101(60)
      abb101(60)=-spbl4k1*abb101(41)
      abb101(44)=spbk2k1*abb101(44)
      abb101(34)=spae1k1*abb101(34)
      abb101(38)=abb101(38)*spae1k1
      abb101(30)=abb101(30)*spae1k1
      abb101(61)=spbk2e2*abb101(53)
      abb101(53)=spbe2e1*abb101(53)
      abb101(28)=spbk1e1*abb101(28)
      abb101(46)=abb101(46)*spbk1e1
      abb101(41)=abb101(41)*spbk1e1
      abb101(62)=spae2k2*abb101(50)
      abb101(50)=spae1e2*abb101(50)
      R2d101=0.0_ki
      rat2 = rat2 + R2d101
      if (debug_nlo_diagrams) then
          write (logfile,*) "<result name='r2' index='101' value='", &
          & R2d101, "'/>"
      end if
   end subroutine
end module p2_gg_httbar_abbrevd101h4_qp
